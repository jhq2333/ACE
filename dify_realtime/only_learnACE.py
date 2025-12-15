#!/usr/bin/env python3
"""
Dify-ACE Learning-Only Integration
ACE 仅用于学习和分析，不生成回复
Dify 的 LLM 继续负责回复用户
"""

import os
import json
import time
import threading
import re
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
from datetime import datetime
from flask import Flask, request, jsonify
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

# ACE Framework - 仅用于学习分析
try:
    from ace import (
        Reflector,
        Curator,
        Playbook,
        LiteLLMClient,
        Sample,
        GeneratorOutput,
        ReflectorOutput,
    )
    from ace.delta import DeltaOperation, DeltaBatch
    ACE_AVAILABLE = True
except ImportError:
    print("⚠️  ACE framework not installed. Run: pip install ace-framework")
    ACE_AVAILABLE = False

# Configuration
DIFY_API_BASE_URL = os.environ.get("DIFY_API_BASE_URL", "http://36.248.221.38")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY", "app-C1b04YCrIQohM1r7z5RfrjjL")

# ACE Configuration
ACE_MODEL = os.environ.get("ACE_MODEL", "openai/qwen-max-latest")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-25587b057d5242428bb940d44035b5fd")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")


# Training Configuration
CACHE_SIZE = int(os.environ.get("CACHE_SIZE","30"))
TRAINING_INTERVAL = int(os.environ.get("TRAINING_INTERVAL","300"))
BACKGROUND_POLL_INTERVAL = int(os.environ.get("BACKGROUND_POLL_INTERVAL","60"))

# Bonus applied when agent requests contact and visitor provides it shortly after
CONTACT_OBTAIN_BONUS = float(os.environ.get("CONTACT_OBTAIN_BONUS", "0.05"))

# Storage
PLAYBOOK_DIR = Path("/data/fangsy/jhq/kouqiangACE/agentic-context-engine-main/dify_realtime/playbooks")
PLAYBOOK_DIR.mkdir(exist_ok=True)
INSIGHTS_DIR = Path("/data/fangsy/jhq/kouqiangACE/agentic-context-engine-main/dify_realtime/insights")
INSIGHTS_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Rotating file handler (INSIGHTS_DIR/only_learnACE.log)
try:
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = INSIGHTS_DIR / "only_learnACE.log"
    rf_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    rf_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    rf_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)

    # attach handlers and avoid double logging by disabling propagation
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(rf_handler)
    logger.propagate = False
except Exception as e:
    # If file handler cannot be set up, continue using console logging
    logger.warning("Failed to initialize RotatingFileHandler: %s", e)

class DialogueQualityAnalyzer:
    """对话质量分析器（基于规则表）"""
    
    def __init__(self):
        # 定义规则库
        self.rules = [
            # S级 - 重要 (扣分 1.0)
            {"code": "S拒诊", "pattern": r"看不了|去别处|不治|治不了|没法看", "desc": "S拒诊: 客服不应直接拒诊", "penalty": 1.0},
            {"code": "S指导用药", "pattern": r"吃.*药|服用|口服.*片|胶囊|使用.*剂", "desc": "S指导用药: 不能指导患者吃某些药物", "penalty": 1.0},
            {"code": "S敏感信息", "pattern": r"最好的|第一|顶级|根治|包治", "desc": "S敏感信息: 涉及广告法敏感词或过度承诺", "penalty": 0.8},
            
            # A级 - 问诊/答疑 (扣分 0.4)
            {"code": "A唯一性诊断", "pattern": r"就是.*病|肯定是|一定是|百分之百", "desc": "A唯一性诊断: 诊断过于绝对", "penalty": 0.4},
            {"code": "A报具体价格", "pattern": r"\d+(元|块|千|万)", "desc": "A报具体价格: 不应直接回答准确费用，应给范围", "penalty": 0.4},
            {"code": "A直接提供缓解方案", "pattern": r"吃醋|姜片|大蒜|偏方", "desc": "A直接提供缓解方案: 不应直接提供非医疗缓解方案", "penalty": 0.4},
            {"code": "A单句问诊过多", "func": lambda q, a: a.count("？") + a.count("?") > 2, "desc": "A单句问诊过多: 一句问话超过2个问题", "penalty": 0.4},
            
            # B级 - 其他 (扣分 0.2)
            {"code": "B字数过长", "func": lambda q, a: len(a) > 35, "desc": "B字数过长: 回复建议控制在50-80字以内", "penalty": 0.2},
            {"code": "B话术不完整", "pattern": r"[，,、]$", "desc": "B话术不完整: 句子似乎未结束", "penalty": 0.2},
        ]
    
    def analyze_dialogue(self, question: str, answer: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        分析对话质量
        返回详细的评分和改进建议
        """
        # 1. 规则检查
        violations = self._check_rule_violations(question, answer)
        
        # 2. 基础指标评估
        metrics = {
            "politeness": self._evaluate_politeness(answer),
            "professionalism": self._evaluate_professionalism(answer),
            "relevance": self._evaluate_relevance(question, answer),
            "clarity": self._evaluate_clarity(answer)
        }
        
        # 3. 计算总分
        # 基础分 1.0，根据违规扣分
        rule_score = 1.0
        for v in violations:
            rule_score -= v["penalty"]
        rule_score = max(0.0, rule_score)
        
        # 综合评分：规则分占 60%，基础指标占 40%
        base_metrics_avg = sum(metrics.values()) / len(metrics)
        overall_score = (rule_score * 0.6) + (base_metrics_avg * 0.4)
        
        metrics["rule_compliance"] = rule_score
        metrics["overall_score"] = overall_score
        
        # 4. 生成改进建议
        suggestions = [v["desc"] for v in violations]
        suggestions.extend(self._generate_suggestions(metrics, question, answer))
        
        # 用户满意度（如果有）
        if metadata and "user_satisfaction" in metadata:
            metrics["user_satisfaction"] = metadata["user_satisfaction"]

        return {
            "metrics": metrics,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }

    def _check_rule_violations(self, question: str, answer: str) -> List[Dict]:
        """检查是否违反规则"""
        violations = []
        for rule in self.rules:
            hit = False
            if "pattern" in rule:
                if re.search(rule["pattern"], answer):
                    hit = True
            elif "func" in rule:
                if rule["func"](question, answer):
                    hit = True
            
            if hit:
                violations.append(rule)
        return violations
    
    def _evaluate_politeness(self, answer: str) -> float:
        """评估礼貌性"""
        score = 0.1
        polite_words = ["您好", "您", "感谢", "请", "抱歉", "对不起", "谢谢"]
        for word in polite_words:
            if word in answer:
                score += 0.08
        
        impolite_words = ["不要", "别", "不懂", "笨", "烦"]
        for word in impolite_words:
            if word in answer:
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_completeness(self, answer: str) -> float:
        """评估简洁性 (已整合到规则检查中，保留此方法用于基础指标)"""
        score = 0.2
        if 5 <= len(answer) <= 25:
            score += 0.2
        elif len(answer) > 50:
            score -= 0.2
        return min(1.0, score)
    
    def _evaluate_contact_info(self, answer: str) -> float:
        """评估是否包含有要到联系方式"""
        score=0.2
        contact_patterns = [r"\d{11}", r"电话[:：]?\w+", r"联系方式[:：]?\d+"]
        for pattern in contact_patterns:
            if re.search(pattern, answer):
                score+=0.2  # 要到联系方式
        return min(1.0, score)
    
    def _evaluate_professionalism(self, answer: str) -> float:
        """评估专业性（口腔医疗）"""
        score = 0.2
        
        # 专业术语
        professional_terms = [
            "口腔", "牙齿", "治疗", "诊断", "预约", "检查",
            "洗牙", "补牙", "拔牙", "矫正", "种植", "根管",
            "牙周", "龋齿", "牙龈", "正畸", "修复"
        ]
        matches = sum(1 for term in professional_terms if term in answer)
        score += min(0.1, matches * 0.08)
        
        # 避免过于口语化
        casual_words = ["哈哈", "呵呵", "嗯嗯", "啊哦", "额"]
        for word in casual_words:
            if word in answer:
                score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """评估相关性"""
        if not question:
            return 0.5
        
        # 关键词重叠
        question_chars = set(question)
        answer_chars = set(answer)
        overlap = len(question_chars & answer_chars)
        relevance = overlap / max(len(question_chars), 1)
        
        return min(1.0, max(0.3, relevance))
    
    def _evaluate_clarity(self, answer: str) -> float:
        """评估清晰度"""
        score = 0.3
        # 句子长度适中
        sentences = re.split(r"[。！？]", answer)
        avg_length = sum(len(s) for s in sentences if s) / max(len(sentences), 1)
        if 10 <= avg_length <= 20:
            score += 0.2
        return min(1.0, score)
    
    def _generate_suggestions(self, metrics: Dict[str, float], question: str, answer: str) -> List[str]:
        """生成基础改进建议"""
        suggestions = []
        
        if metrics["overall_score"] < 0.6:
            suggestions.append("🔴 整体质量需要显著改进")
        elif metrics["overall_score"] < 0.75:
            suggestions.append("🟡 整体质量尚可，有改进空间")
        else:
            suggestions.append("🟢 整体质量良好")
        
        if metrics["professionalism"] < 0.4:
            suggestions.append("建议：使用更多专业医疗术语，提升专业感") 
        # 针对性建议
        if "预约" in question and "预约" not in answer:
            suggestions.append("⚠️  用户询问预约，但回答中未提及预约相关信息")
        
        return suggestions


class ACELearningManager:
    """ACE 学习管理器（仅学习，不生成回复）"""
    
    def __init__(self, task_type: str = "customer_service"):
        self.task_type = task_type
        self.playbook_path = PLAYBOOK_DIR / f"{task_type}_insights.json"
        self.analyzer = DialogueQualityAnalyzer()
        
        if not ACE_AVAILABLE or not OPENAI_API_KEY:
            print("⚠️  ACE not fully available, using analysis-only mode")
            self.reflector = None
            self.curator = None
            self.playbook = None
            return
        
        try:
            self.llm = LiteLLMClient(
                model="openai/qwen-max-latest",
                api_key="sk-25587b057d5242428bb940d44035b5fd",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.3,
                max_tokens=4000,
                timeout=60
            )
            
            # 加载或创建 Playbook
            if self.playbook_path.exists():
                print(f"📚 Loading insights from {self.playbook_path}")
                self.playbook = Playbook.load_from_file(str(self.playbook_path))
            else:
                print(f"📚 Creating new insights playbook")
                self.playbook = Playbook()
            
            # 只需要 Reflector 和 Curator 来学习
            self.reflector = Reflector(self.llm)
            self.curator = Curator(self.llm)
            
            print(f"✅ ACE Learning Manager initialized")
            print(f"   Current insights: {len(self.playbook.bullets())}")
            
        except Exception as e:
            print(f"❌ Failed to initialize ACE: {e}")
            self.reflector = None
            self.curator = None
    
    def learn_from_dialogues(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从对话中学习（不生成回复）
        分析质量 → 反思 → 更新知识库
        """
        if not self.reflector or not self.curator:
            return self._fallback_analysis(dialogues)
        
        try:
            print(f"\n{'='*60}")
            print(f"📖 Learning from Dialogues")
            print(f"{'='*60}")
            print(f"Dialogues: {len(dialogues)}")
            
            start_time = time.time()
            insights = []
            
            for dialogue in dialogues:
                messages = dialogue.get("messages", [])
                session_id = dialogue.get("session_id", "unknown")
                
                # 提取问答对
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "visitor" and messages[i + 1].get("role") == "agent":
                        question = messages[i].get("content", "").strip()
                        answer = messages[i + 1].get("content", "").strip()
                        
                        if question and answer:
                            # 分析质量
                            analysis = self.analyzer.analyze_dialogue(
                                question, 
                                answer,
                                messages[i].get("metadata")
                            )
                            
                            # Detect if agent asked for contact and visitor provided it shortly after
                            asked_for_contact = False
                            contact_request_patterns = [
                                r"请问.*(联系方式|手机号|电话)",
                                r"(方便留|麻烦留|能留|留下).*(联系方式|手机号|电话)",
                                r"(留下联系方式|要联系方式|要手机号|留下手机号)",
                                r"(可以留一下联系方式|能留一下联系方式)"
                            ]
                            for p in contact_request_patterns:
                                if re.search(p, answer):
                                    asked_for_contact = True
                                    break

                            contact_obtained = False
                            # look for a visitor message after the agent reply (within next 3 messages)
                            for j in range(i + 2, min(i + 6, len(messages))):
                                if messages[j].get("role") == "visitor":
                                    content_j = messages[j].get("content", "")
                                    if re.search(r"\d{11}", content_j) or re.search(r"(联系方式[:：]?\s*\d+)|((手机号|电话)[:：]?\s*\d+)", content_j):
                                        contact_obtained = True
                                        break

                            if asked_for_contact and contact_obtained:
                                analysis.setdefault("metrics", {})
                                analysis["metrics"]["contact_obtained"] = True
                                bonus = CONTACT_OBTAIN_BONUS
                                if "overall_score" in analysis["metrics"]:
                                    prev = analysis["metrics"]["overall_score"]
                                    analysis["metrics"]["overall_score"] = min(1.0, prev + bonus)
                                else:
                                    analysis["metrics"]["overall_score"] = min(1.0, bonus)
                                analysis.setdefault("suggestions", []).append("✅ 成功获取联系方式，给予评分加分")

                            # 使用 Reflector 反思（基于 Dify 的实际回答）
                            system_prompt = messages[i].get("metadata", {}).get("system_prompt")
                            # Short audit log so we can trace which QA triggered reflection
                            logger.info("Reflecting on QA - session=%s q_index=%d question=%s", session_id, i, question[:80])
                            reflection = self._reflect_on_answer(
                                question,
                                answer,
                                analysis,
                                system_prompt=system_prompt,
                            )
                            logger.debug("analysis: %s", analysis)
                            # 使用 Curator 更新知识库
                            if reflection:
                                self._update_playbook(reflection, question, analysis)
                            
                            insights.append({
                                "session_id": session_id,
                                "question": question[:50] + "...",
                                "quality_score": analysis["metrics"]["overall_score"],
                                "suggestions": analysis["suggestions"]
                            })
            
            # 保存更新后的知识库
            if self.playbook:
                self.playbook.save_to_file(str(self.playbook_path))
            
            learning_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"✅ Learning Completed")
            print(f"{'='*61}")
            print(f"Time: {learning_time:.2f}s")
            print(f"Dialogues analyzed: {len(insights)}")
            print(f"Knowledge base size: {len(self.playbook.bullets())}")
            
            return {
                "status": "success",
                "learning_time": learning_time,
                "dialogues_analyzed": len(insights),
                "insights_count": len(self.playbook.bullets()),
                "insights": insights[:5]  # 返回前5个示例
            }
            
        except Exception as e:
            print(f"❌ Learning failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _reflect_on_answer(self, question: str, answer: str, analysis: Dict, system_prompt: str = None) -> Optional[Any]:
        """使用 Reflector 反思 Dify 的回答"""
        try:
            # Build a GeneratorOutput-like object representing the observed answer
            # so the Reflector can analyze the generator's trajectory.
            gen_reasoning = "\n".join(analysis.get('suggestions', [])) or "(no reasoning provided)"
            gen_out = GeneratorOutput(
                reasoning=gen_reasoning,
                final_answer=answer,
                bullet_ids=[],
                raw={"answer": answer, "analysis": analysis},
            )

            # Call the Reflector to produce a structured reflection
            # If system prompt is provided, include it in the question context
            reflect_question = question
            if system_prompt:
                reflect_question = f"{question}\n\n[System Prompt/Context]:\n{system_prompt}"

            reflector_output = self.reflector.reflect(
                question=reflect_question,
                generator_output=gen_out,
                playbook=self.playbook,
                ground_truth=None,
                feedback=f"Quality:{analysis['metrics']['overall_score']:.2%}",
            )

            if gen_out.reasoning:
                logger.debug("思考过程: %s", gen_out.reasoning[:1000])

            # Safely access reflector fields and log them (truncate long texts)
            ro_reasoning = getattr(reflector_output, 'reasoning', None)
            if ro_reasoning:
                logger.info("反思分析: %s", str(ro_reasoning)[:1000])
                logger.info("错误定位: %s", str(getattr(reflector_output, 'error_identification', ''))[:500])
                logger.info("根本原因分析: %s", str(getattr(reflector_output, 'root_cause_analysis', ''))[:500])
                logger.info("正确的处理方法: %s", str(getattr(reflector_output, 'correct_approach', ''))[:500])
                logger.info("关键洞察: %s", str(getattr(reflector_output, 'key_insight', ''))[:500])

            return reflector_output
            
        except Exception as e:
            logger.exception("Reflection error")
            return None
    
    def _update_playbook(self, reflection: Dict, question: str, analysis: Dict):
        """更新知识库"""
        try:
            # If we have a ReflectorOutput, use Curator to produce a delta and apply it
            if reflection is not None and hasattr(self, 'curator') and self.curator:
                try:
                    progress = f"quality_score={analysis['metrics']['overall_score']:.2%}"
                    curator_output = self.curator.curate(
                        reflection=reflection,
                        playbook=self.playbook,
                        question_context=self.task_type,
                        progress=progress,
                    )

                    # Apply the delta produced by the curator, with a brief summary log
                    if curator_output and getattr(curator_output, 'delta', None) is not None:
                        try:
                            delta = curator_output.delta
                            ops = getattr(delta, 'operations', []) or []
                            before_count = len(self.playbook.bullets()) if self.playbook else 0

                            # summarize ops
                            added = [o.bullet_id or '<new>' for o in ops if o.type.upper() == 'ADD']
                            updated = [o.bullet_id for o in ops if o.type.upper() == 'UPDATE' and o.bullet_id]
                            removed = [o.bullet_id for o in ops if o.type.upper() == 'REMOVE' and o.bullet_id]

                            print(f"Curator produced delta: total_ops={len(ops)} ADD={len(added)} UPDATE={len(updated)} REMOVE={len(removed)}")
                            if added:
                                print("  Added (ids or <new>):", added)
                            if updated:
                                print("  Updated (ids):", updated)
                            if removed:
                                print("  Removed (ids):", removed)

                            # apply delta
                            self.playbook.apply_delta(delta)

                            # Run lightweight semantic merge post-processing to consolidate similar bullets
                            try:
                                merge_result = self._semantic_merge_postprocess(threshold=0.75)
                                merged = merge_result.get('merged', 0)
                                removed_ids = merge_result.get('removed', [])
                            except Exception as e:
                                print(f"Semantic merge failed: {e}")
                                merged = 0
                                removed_ids = []

                            after_count = len(self.playbook.bullets()) if self.playbook else 0
                            print(f"Playbook size: before={before_count} after={after_count} (delta={after_count-before_count}) merged={merged}")
                            if removed_ids:
                                print("  Removed merged ids:", removed_ids)

                        except Exception as e:
                            print(f"Error applying curator delta: {e}")

                except Exception as e:
                    print(f"Curator error: {e}")

            # Preserve the lightweight insight-adding behavior as a fallback/augmentation
            if analysis["metrics"]["overall_score"] >= 0.75:
                insight = f"✅ 优质回答示例：{question} → 质量分: {analysis['metrics']['overall_score']:.2%}"
                self._add_insight(insight, helpful=True)

            elif analysis["metrics"]["overall_score"] < 0.6:
                for suggestion in analysis["suggestions"]:
                    if suggestion.startswith("建议") or suggestion.startswith("⚠️"):
                        self._add_insight(suggestion, helpful=False)
        
        except Exception as e:
            print(f"Playbook update error: {e}")
    
    def _add_insight(self, content: str, helpful: bool = True):
        """添加洞察到知识库 (覆盖相似/冲突规则)"""
        if not self.playbook:
            return
        
        bullets = self.playbook.bullets()
        for b in bullets:
            content_old = b.content or ""
            
            # 1. 检查规则代码前缀 (例如 "S拒诊:")
            code_new = content.split(":")[0].strip() if ":" in content else ""
            code_old = content_old.split(":")[0].strip() if ":" in content_old else ""
            
            is_rule_conflict = False
            # 只有当看起来像规则代码时才匹配 (长度>1, 且相等)
            # 排除 "建议" 这种通用前缀
            if code_new and code_new == code_old and len(code_new) > 1 and code_new != "建议":
                is_rule_conflict = True
            
            # 2. 检查语义相似度
            similarity = SequenceMatcher(None, content_old, content).ratio()
            
            if is_rule_conflict or similarity > 0.80:
                # 覆盖现有规则
                current_helpful = getattr(b, "helpful_count", 0)
                current_harmful = getattr(b, "harmful_count", 0)
                
                self.playbook.update_bullet(
                    b.id,
                    content=content, # 使用新内容覆盖
                    metadata={
                        "helpful": current_helpful + (1 if helpful else 0),
                        "harmful": current_harmful + (0 if helpful else 1),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                print(f"🔄 Updated insight: {content_old[:20]}... -> {content[:20]}...")
                return

        # 新增
        self.playbook.add_bullet(
            section="客服质量改进",
            content=content,
            metadata={
                "helpful": 1 if helpful else 0,
                "harmful": 0 if helpful else 1,
                "timestamp": datetime.now().isoformat()
            }
        )

    def _semantic_merge_postprocess(self, threshold: float = 0.75) -> Dict[str, Any]:
        """
        Lightweight semantic merge using difflib.SequenceMatcher.

        - For each section, compares bullet contents pairwise and when similarity
          >= threshold merges the second bullet into the first (keeps the first).
        - Merging strategy: keep the longer content and sum helpful/harmful/neutral counts.
        - Returns a dict with number merged and list of removed ids.
        """
        result = {"merged": 0, "removed": []}
        if not self.playbook:
            return result

        try:
            # Access sections (internal structure); keep operations small so O(n^2) is acceptable for small playbooks
            sections = list(getattr(self.playbook, "_sections", {}).keys())
            for section in sections:
                ids = list(getattr(self.playbook, "_sections", {}).get(section, []))
                # compare pairwise
                i = 0
                while i < len(ids):
                    id_i = ids[i]
                    b_i = self.playbook.get_bullet(id_i)
                    if b_i is None:
                        i += 1
                        continue
                    j = i + 1
                    while j < len(ids):
                        id_j = ids[j]
                        b_j = self.playbook.get_bullet(id_j)
                        if b_j is None:
                            j += 1
                            continue
                        sim = SequenceMatcher(None, (b_i.content or ""), (b_j.content or "")).ratio()
                        if sim >= threshold:
                            # Merge b_j into b_i: choose longer content and sum counters
                            new_content = b_i.content if len(b_i.content or "") >= len(b_j.content or "") else b_j.content
                            new_meta = {
                                "helpful": (getattr(b_i, "helpful", 0) or 0) + (getattr(b_j, "helpful", 0) or 0),
                                "harmful": (getattr(b_i, "harmful", 0) or 0) + (getattr(b_j, "harmful", 0) or 0),
                                "neutral": (getattr(b_i, "neutral", 0) or 0) + (getattr(b_j, "neutral", 0) or 0),
                            }
                            try:
                                self.playbook.update_bullet(id_i, content=new_content, metadata=new_meta)
                                self.playbook.remove_bullet(id_j)
                            except Exception:
                                # best-effort: if update/remove fails skip
                                j += 1
                                continue
                            result["merged"] += 1
                            result["removed"].append(id_j)
                            # remove id_j from our local list so we don't re-check it
                            ids.pop(j)
                            # don't increment j, check next item now at position j
                        else:
                            j += 1
                    i += 1
        except Exception as e:
            print(f"Semantic merge error: {e}")

        return result
    
    def _fallback_analysis(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """降级方案：仅使用规则分析"""
        insights = []
        
        for dialogue in dialogues:
            messages = dialogue.get("messages", [])
            
            for i in range(len(messages) - 1):
                if messages[i].get("role") == "visitor" and messages[i + 1].get("role") == "agent":
                    question = messages[i].get("content", "")
                    answer = messages[i + 1].get("content", "")
                    
                    if question and answer:
                        analysis = self.analyzer.analyze_dialogue(question, answer)
                        insights.append({
                            "question": question[:50] + "...",
                            "quality_score": analysis["metrics"]["overall_score"],
                            "suggestions": analysis["suggestions"]
                        })
        
        return {
            "status": "success",
            "mode": "rule_based_only",
            "dialogues_analyzed": len(insights),
            "insights": insights[:10]
        }
    
    def get_insights(self) -> Dict[str, Any]:
        """获取学习到的洞察"""
        if not self.playbook:
            return {"insights": [], "count": 0}
        
        bullets = self.playbook.bullets()
        
        return {
            "count": len(bullets),
            "insights": [
                {
                    "content": b.content,
                    "helpful": b.helpful_count,
                    "harmful": b.harmful_count
                }
                for b in bullets[:20]  # 返回前20条
            ]
        }


class DifyACEIntegration:
    """Dify-ACE 集成（仅学习模式）"""
    
    def __init__(self):
        self.dialogue_cache: List[Dict[str, Any]] = []
        self.last_learning_time = 0
        self.learners: Dict[str, ACELearningManager] = {}
        
        # 初始化默认学习器
        self._get_learner("customer_service")
    
    def _get_learner(self, task_type: str) -> ACELearningManager:
        """获取或创建学习器"""
        if task_type not in self.learners:
            self.learners[task_type] = ACELearningManager(task_type)
        return self.learners[task_type]
    
    def add_dialogue(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """添加对话"""
        self.dialogue_cache.append({
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        if len(self.dialogue_cache) > CACHE_SIZE * 3:
            self.dialogue_cache = self.dialogue_cache[-CACHE_SIZE * 2:]
        
        print(f"📝 Dialogue added: {role} in {session_id} (cache: {len(self.dialogue_cache)})")
    
    def should_learn(self) -> bool:
        """判断是否应该触发学习"""
        cache_full = len(self.dialogue_cache) >= CACHE_SIZE
        time_passed = (time.time() - self.last_learning_time) >= TRAINING_INTERVAL
        return cache_full and time_passed
    
    def format_dialogues_for_learning(self) -> List[Dict[str, Any]]:
        """格式化对话为学习数据"""
        by_session = {}
        for d in self.dialogue_cache:
            sid = d.get("session_id", "unknown")
            by_session.setdefault(sid, []).append({
                "role": d.get("role"),
                "content": d.get("content"),
                "metadata": d.get("metadata", {})
            })
        
        dialogues = []
        for sid, msgs in by_session.items():
            dialogues.append({
                "session_id": sid,
                "messages": msgs,
                "timestamp": datetime.now().isoformat()
            })
        
        return dialogues
    
    def trigger_learning(self, task_type: str = "customer_service") -> Dict[str, Any]:
        """触发学习"""
        try:
            dialogues = self.format_dialogues_for_learning()
            
            if not dialogues:
                return {"error": "No dialogues to learn from"}
            
            print(f"\n🎓 Triggering learning for {task_type}")
            print(f"   Dialogues: {len(dialogues)}")
            
            learner = self._get_learner(task_type)
            result = learner.learn_from_dialogues(dialogues)
            
            if result.get("status") == "success":
                self.last_learning_time = time.time()
                self.dialogue_cache = []
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def get_insights(self, task_type: str = "customer_service") -> Dict[str, Any]:
        """获取学习洞察"""
        learner = self._get_learner(task_type)
        return learner.get_insights()


# Flask app
app = Flask(__name__)
integration = DifyACEIntegration()
stop_event = threading.Event()


def background_learning_loop(poll_interval: int = BACKGROUND_POLL_INTERVAL):
    """后台学习循环"""
    print(f"🔄 Background learner started (poll interval: {poll_interval}s)")
    
    while not stop_event.is_set():
        try:
            if integration.should_learn():
                print("\n" + "="*60)
                print("🤖 Background: Triggering automatic learning")
                print("="*60)
                
                result = integration.trigger_learning()
                print(f"Learning result: {result.get('status')}")
                
        except Exception as e:
            # Log error and continue
            print(f"❌ Background learner error: {e}")
        
        stop_event.wait(poll_interval)


def _start_background_learner():
    """启动后台学习线程"""
    t = threading.Thread(target=background_learning_loop, daemon=True)
    t.start()


# Flask Routes
@app.route("/webhook", methods=["POST"])
def webhook():
    """接收 Dify webhook"""
    try:
        data = request.json or {}
        session_id = data.get("conversation_id") or data.get("session_id") or "unknown"
        
        # 提取用户评分
        user_satisfaction = None
        if "feedback" in data:
            feedback = data.get("feedback", {})
            if isinstance(feedback, dict):
                user_satisfaction = feedback.get("rating")
        
        # 提取系统提示词（如果有）
        system_prompt = data.get("prompt") or data.get("system_prompt")
        
        # 添加对话
        if "query" in data:
            integration.add_dialogue(
                session_id, 
                "visitor", 
                data.get("query"),
                {
                    "user_satisfaction": user_satisfaction,
                    "system_prompt": system_prompt
                }
            )
        
        if "answer" in data:
            integration.add_dialogue(session_id, "agent", data.get("answer"))
        
        return jsonify({
            "status": "ok",
            "cached": len(integration.dialogue_cache),
            "next_learning_in": max(0, TRAINING_INTERVAL - (time.time() - integration.last_learning_time))
        })
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/manual_learn", methods=["POST"])
def manual_learn():
    """手动触发学习"""
    try:
        data = request.json or {}
        task_type = data.get("task_type", "customer_service")
        
        print(f"\n🎯 Manual learning triggered for {task_type}")
        
        result = integration.trigger_learning(task_type=task_type)
        insights = integration.get_insights(task_type=task_type)
        
        return jsonify({
            "learning_result": result,
            "insights": insights
        })
        
    except Exception as e:
        print(f"❌ Manual learn error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/insights/<task_type>", methods=["GET"])
def get_insights(task_type):
    """获取学习洞察"""
    try:
        insights = integration.get_insights(task_type)
        return jsonify(insights)
    except Exception as e:
        print(f"❌ Insights error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    """系统状态"""
    return jsonify({
        "status": "running",
        "mode": "learning_only",
        "ace_available": ACE_AVAILABLE,
        "model": ACE_MODEL,
        "dialogue_cache_size": len(integration.dialogue_cache),
        "cache_threshold": CACHE_SIZE,
        "last_learning": integration.last_learning_time,
        "next_learning_in": max(0, TRAINING_INTERVAL - (time.time() - integration.last_learning_time))
    })



if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 Dify-ACE Learning-Only Integration")
    print("="*60)
    print(f"User: jhq2333")
    print(f"Server: 123.181.192.120:18758")
    print(f"Mode: Learning Only (No Generation)")
    print(f"ACE Available: {ACE_AVAILABLE}")
    print(f"Model: {ACE_MODEL}")
    print(f"Cache Threshold: {CACHE_SIZE} dialogues")
    print("="*60 + "\n")
    
    _start_background_learner()
    app.run(host="0.0.0.0", port=18578, debug=False)
