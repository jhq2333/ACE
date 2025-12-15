#!/usr/bin/env python3
"""Combined Dify webhook learner + human review learner sharing one Flask app."""

import json
import logging
import os
import re
import threading
import time
import traceback
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from logging.handlers import RotatingFileHandler

try:
    from ace import (
        Curator,
        GeneratorOutput,
        LiteLLMClient,
        Playbook,
        Reflector,
    )
    ACE_AVAILABLE = True
except ImportError:
    print("⚠️  ACE framework not installed. Run: pip install ace-framework")
    ACE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ACE_MODEL = os.environ.get("ACE_MODEL", "openai/qwen-max-latest")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-25587b057d5242428bb940d44035b5fd")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "30"))
TRAINING_INTERVAL = int(os.environ.get("TRAINING_INTERVAL", "300"))
BACKGROUND_POLL_INTERVAL = int(os.environ.get("BACKGROUND_POLL_INTERVAL", "60"))
CONTACT_OBTAIN_BONUS = float(os.environ.get("CONTACT_OBTAIN_BONUS", "0.05"))

ROOT_DIR = Path("/data/fangsy/jhq/kouqiangACE/agentic-context-engine-main/dify_realtime")
PLAYBOOK_DIR = ROOT_DIR / "playbooks"
PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
INSIGHTS_DIR = ROOT_DIR / "insights"
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_LOG_DIR = ROOT_DIR / "reviews"
REVIEW_LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = Path(os.environ.get("REVIEW_DATA_FILE", "review_data.json")).resolve()
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
if not DATA_FILE.exists():
    DATA_FILE.write_text("[]", encoding="utf-8")

LOG_FILE = INSIGHTS_DIR / "combined_learning.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("combined_learning")
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    rf_handler = RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    rf_handler.setLevel(logging.DEBUG)
    rf_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(rf_handler)

# ---------------------------------------------------------------------------
# ACE shared components
# ---------------------------------------------------------------------------


class ACEComponents:
    """Bundle of ACE resources shared by auto-learning and human review."""

    def __init__(self, task_type: str):
        self.task_type = task_type
        self.playbook_path = PLAYBOOK_DIR / f"{task_type}_insights.json"
        self.lock = threading.Lock()
        self.llm: Optional[LiteLLMClient] = None
        self.playbook: Optional[Playbook] = None
        self.reflector: Optional[Reflector] = None
        self.curator: Optional[Curator] = None
        self.ready = False
        self._initialize()

    def _initialize(self) -> None:
        if not ACE_AVAILABLE or not OPENAI_API_KEY:
            logger.warning("ACE components unavailable; running in analysis-only mode")
            return

        try:
            self.llm = LiteLLMClient(
                model=ACE_MODEL,
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_API_BASE,
                temperature=0.3,
                max_tokens=4000,
                timeout=60,
            )

            if self.playbook_path.exists():
                logger.info("📚 Loading playbook %s", self.playbook_path)
                self.playbook = Playbook.load_from_file(str(self.playbook_path))
            else:
                logger.info("📚 Creating new playbook at %s", self.playbook_path)
                self.playbook = Playbook()

            self.reflector = Reflector(self.llm)
            self.curator = Curator(self.llm)
            self.ready = True
            logger.info("✅ ACE components ready for task_type=%s", self.task_type)
        except Exception as exc:
            logger.error("❌ Failed to initialize ACE components: %s", exc)
            self.ready = False


class ACEComponentsRegistry:
    """Lazy factory for ACEComponents keyed by task type."""

    def __init__(self):
        self._components: Dict[str, ACEComponents] = {}
        self._lock = threading.Lock()

    def get(self, task_type: str) -> ACEComponents:
        with self._lock:
            if task_type not in self._components:
                self._components[task_type] = ACEComponents(task_type)
            return self._components[task_type]


# ---------------------------------------------------------------------------
# Auto-learning pipeline
# ---------------------------------------------------------------------------


class DialogueQualityAnalyzer:
    """Rule-based dialogue quality analyzer."""

    def __init__(self):
        self.rules = [
            {"code": "S拒诊", "pattern": r"看不了|去别处|不治|治不了|没法看", "desc": "S拒诊: 客服不应直接拒诊", "penalty": 1.0},
            {"code": "S指导用药", "pattern": r"吃.*药|服用|口服.*片|胶囊|使用.*剂", "desc": "S指导用药: 不能指导患者吃某些药物", "penalty": 1.0},
            {"code": "S敏感信息", "pattern": r"最好的|第一|顶级|根治|包治", "desc": "S敏感信息: 涉及广告法敏感词或过度承诺", "penalty": 0.8},
            {"code": "A唯一性诊断", "pattern": r"就是.*病|肯定是|一定是|百分之百", "desc": "A唯一性诊断: 诊断过于绝对", "penalty": 0.4},
            {"code": "A报具体价格", "pattern": r"\d+(元|块|千|万)", "desc": "A报具体价格: 不应直接回答准确费用，应给范围", "penalty": 0.4},
            {"code": "A直接提供缓解方案", "pattern": r"吃醋|姜片|大蒜|偏方", "desc": "A直接提供缓解方案: 不应直接提供非医疗缓解方案", "penalty": 0.4},
            {"code": "A单句问诊过多", "func": lambda q, a: a.count("？") + a.count("?") > 2, "desc": "A单句问诊过多: 一句问话超过2个问题", "penalty": 0.4},
            {"code": "B字数过长", "func": lambda q, a: len(a) > 35, "desc": "B字数过长: 回复建议控制在50-80字以内", "penalty": 0.2},
            {"code": "B话术不完整", "pattern": r"[，,、]$", "desc": "B话术不完整: 句子似乎未结束", "penalty": 0.2},
        ]

    def analyze_dialogue(self, question: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        violations = self._check_rule_violations(question, answer)
        metrics = {
            "politeness": self._evaluate_politeness(answer),
            "professionalism": self._evaluate_professionalism(answer),
            "relevance": self._evaluate_relevance(question, answer),
            "clarity": self._evaluate_clarity(answer),
        }

        rule_score = 1.0
        for violation in violations:
            rule_score -= violation["penalty"]
        rule_score = max(0.0, rule_score)

        base_avg = sum(metrics.values()) / len(metrics)
        overall_score = (rule_score * 0.6) + (base_avg * 0.4)
        metrics["rule_compliance"] = rule_score
        metrics["overall_score"] = overall_score

        suggestions = [v["desc"] for v in violations]
        suggestions.extend(self._generate_suggestions(metrics, question, answer))

        if metadata and "user_satisfaction" in metadata:
            metrics["user_satisfaction"] = metadata["user_satisfaction"]

        return {
            "metrics": metrics,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat(),
        }

    def _check_rule_violations(self, question: str, answer: str) -> List[Dict[str, Any]]:
        hits = []
        for rule in self.rules:
            trigger = False
            if "pattern" in rule and re.search(rule["pattern"], answer):
                trigger = True
            elif "func" in rule and rule["func"](question, answer):
                trigger = True
            if trigger:
                hits.append(rule)
        return hits

    @staticmethod
    def _evaluate_politeness(answer: str) -> float:
        score = 0.1
        polite_words = ["您好", "您", "感谢", "请", "抱歉", "对不起", "谢谢"]
        impolite_words = ["不要", "别", "不懂", "笨", "烦"]
        for word in polite_words:
            if word in answer:
                score += 0.08
        for word in impolite_words:
            if word in answer:
                score -= 0.3
        return max(0.0, min(1.0, score))

    @staticmethod
    def _evaluate_professionalism(answer: str) -> float:
        score = 0.2
        professional_terms = [
            "口腔",
            "牙齿",
            "治疗",
            "诊断",
            "预约",
            "检查",
            "洗牙",
            "补牙",
            "拔牙",
            "矫正",
            "种植",
            "根管",
            "牙周",
            "龋齿",
            "牙龈",
            "正畸",
            "修复",
        ]
        matches = sum(1 for term in professional_terms if term in answer)
        score += min(0.1, matches * 0.08)
        for word in ["哈哈", "呵呵", "嗯嗯", "啊哦", "额"]:
            if word in answer:
                score -= 0.15
        return max(0.0, min(1.0, score))

    @staticmethod
    def _evaluate_relevance(question: str, answer: str) -> float:
        if not question:
            return 0.5
        question_chars = set(question)
        answer_chars = set(answer)
        overlap = len(question_chars & answer_chars)
        relevance = overlap / max(len(question_chars), 1)
        return min(1.0, max(0.3, relevance))

    @staticmethod
    def _evaluate_clarity(answer: str) -> float:
        score = 0.3
        sentences = re.split(r"[。！？]", answer)
        valid_sentences = [s for s in sentences if s]
        if valid_sentences:
            avg_len = sum(len(s) for s in valid_sentences) / len(valid_sentences)
            if 10 <= avg_len <= 20:
                score += 0.2
        return min(1.0, score)

    def _generate_suggestions(self, metrics: Dict[str, float], question: str, answer: str) -> List[str]:
        tips: List[str] = []
        if metrics["overall_score"] < 0.6:
            tips.append("🔴 整体质量需要显著改进")
        elif metrics["overall_score"] < 0.75:
            tips.append("🟡 整体质量尚可，有改进空间")
        else:
            tips.append("🟢 整体质量良好")
        if metrics["professionalism"] < 0.4:
            tips.append("建议：使用更多专业医疗术语，提升专业感")
        if "预约" in question and "预约" not in answer:
            tips.append("⚠️  用户询问预约，但回答中未提及预约相关信息")
        return tips


class ACELearningManager:
    """Learns automatically from Dify dialogues."""

    def __init__(self, components: ACEComponents):
        self.components = components
        self.task_type = components.task_type
        self.playbook = components.playbook
        self.reflector = components.reflector
        self.curator = components.curator
        self.analyzer = DialogueQualityAnalyzer()

    def learn_from_dialogues(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.components.ready or not self.playbook or not self.reflector or not self.curator:
            return self._fallback_analysis(dialogues)

        try:
            logger.info("📖 Learning from %d dialogues", len(dialogues))
            start = time.time()
            insights = []

            for dialogue in dialogues:
                messages = dialogue.get("messages", [])
                session_id = dialogue.get("session_id", "unknown")

                for idx in range(len(messages) - 1):
                    cur_msg = messages[idx]
                    nxt_msg = messages[idx + 1]
                    if cur_msg.get("role") != "visitor" or nxt_msg.get("role") != "agent":
                        continue
                    question = cur_msg.get("content", "").strip()
                    answer = nxt_msg.get("content", "").strip()
                    if not question or not answer:
                        continue

                    analysis = self.analyzer.analyze_dialogue(question, answer, cur_msg.get("metadata"))
                    if self._contact_bonus_applies(answer, messages, idx):
                        self._apply_contact_bonus(analysis)

                    logger.info("Reflecting session=%s idx=%d", session_id, idx)
                    reflection = self._reflect_on_answer(question, answer, analysis, system_prompt=cur_msg.get("metadata", {}).get("system_prompt"))
                    if reflection:
                        self._update_playbook(reflection, question, analysis)

                    insights.append(
                        {
                            "session_id": session_id,
                            "question": question[:50] + "...",
                            "quality_score": analysis["metrics"].get("overall_score", 0.0),
                            "suggestions": analysis["suggestions"],
                        }
                    )

            self._save_playbook()
            elapsed = time.time() - start
            logger.info("✅ Auto learning finished in %.2fs", elapsed)
            return {
                "status": "success",
                "learning_time": elapsed,
                "dialogues_analyzed": len(insights),
                "insights_count": len(self.playbook.bullets()) if self.playbook else 0,
                "insights": insights[:5],
            }
        except Exception as exc:
            logger.error("❌ Learning failed: %s", exc)
            traceback.print_exc()
            return {"error": str(exc)}

    @staticmethod
    def _contact_bonus_applies(answer: str, messages: List[Dict[str, Any]], idx: int) -> bool:
        patterns = [
            r"请问.*(联系方式|手机号|电话)",
            r"(方便留|麻烦留|能留|留下).*(联系方式|手机号|电话)",
            r"(留下联系方式|要联系方式|要手机号|留下手机号)",
            r"(可以留一下联系方式|能留一下联系方式)",
        ]
        asked = any(re.search(p, answer) for p in patterns)
        if not asked:
            return False
        for follow_idx in range(idx + 2, min(idx + 6, len(messages))):
            msg = messages[follow_idx]
            if msg.get("role") != "visitor":
                continue
            text = msg.get("content", "")
            if re.search(r"\d{11}", text) or re.search(r"(联系方式|电话|手机号)[:：]?\s*\d+", text):
                return True
        return False

    @staticmethod
    def _apply_contact_bonus(analysis: Dict[str, Any]) -> None:
        metrics = analysis.setdefault("metrics", {})
        prev = metrics.get("overall_score", 0.0)
        metrics["overall_score"] = min(1.0, prev + CONTACT_OBTAIN_BONUS)
        analysis.setdefault("suggestions", []).append("✅ 成功获取联系方式，给予评分加分")
        metrics["contact_obtained"] = True

    def _reflect_on_answer(self, question: str, answer: str, analysis: Dict[str, Any], system_prompt: Optional[str] = None) -> Optional[Any]:
        if not self.reflector or not self.playbook:
            return None
        try:
            reasoning = "\n".join(analysis.get("suggestions", [])) or "(no reasoning provided)"
            generator_output = GeneratorOutput(
                reasoning=reasoning,
                final_answer=answer,
                bullet_ids=[],
                raw={"answer": answer, "analysis": analysis},
            )
            reflect_question = question
            if system_prompt:
                reflect_question = f"{question}\n\n[System Prompt/Context]:\n{system_prompt}"
            return self.reflector.reflect(
                question=reflect_question,
                generator_output=generator_output,
                playbook=self.playbook,
                ground_truth=None,
                feedback=f"Quality:{analysis['metrics'].get('overall_score', 0):.2%}",
            )
        except Exception:
            logger.exception("Reflection error")
            return None

    def _update_playbook(self, reflection: Any, question: str, analysis: Dict[str, Any]) -> None:
        if not self.curator or not self.playbook:
            return
        try:
            progress = f"quality_score={analysis['metrics'].get('overall_score', 0):.2%}"
            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=self.task_type,
                progress=progress,
            )
            if curator_output and getattr(curator_output, "delta", None):
                with self.components.lock:
                    self.playbook.apply_delta(curator_output.delta)
                    self._semantic_merge_postprocess()
            self._add_lightweight_insight(question, analysis)
        except Exception as exc:
            logger.error("Playbook update error: %s", exc)

    def _add_lightweight_insight(self, question: str, analysis: Dict[str, Any]) -> None:
        if not self.playbook:
            return
        score = analysis["metrics"].get("overall_score", 0)
        if score >= 0.75:
            content = f"✅ 优质回答示例：{question} → 质量分: {score:.2%}"
            self._add_insight(content, helpful=True)
        elif score < 0.6:
            for suggestion in analysis.get("suggestions", []):
                if suggestion.startswith("建议") or suggestion.startswith("⚠️"):
                    self._add_insight(suggestion, helpful=False)

    def _add_insight(self, content: str, helpful: bool = True) -> None:
        if not self.playbook:
            return
        with self.components.lock:
            for bullet in self.playbook.bullets():
                old_content = bullet.content or ""
                code_new = content.split(":")[0].strip() if ":" in content else ""
                code_old = old_content.split(":")[0].strip() if ":" in old_content else ""
                conflict = code_new and code_new == code_old and code_new not in {"建议"}
                similarity = SequenceMatcher(None, old_content, content).ratio()
                if conflict or similarity > 0.80:
                    helpful_count = getattr(bullet, "helpful_count", 0)
                    harmful_count = getattr(bullet, "harmful_count", 0)
                    self.playbook.update_bullet(
                        bullet.id,
                        content=content,
                        metadata={
                            "helpful": helpful_count + (1 if helpful else 0),
                            "harmful": harmful_count + (0 if helpful else 1),
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    return
            self.playbook.add_bullet(
                section="客服质量改进",
                content=content,
                metadata={
                    "helpful": 1 if helpful else 0,
                    "harmful": 0 if helpful else 1,
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def _semantic_merge_postprocess(self, threshold: float = 0.75) -> None:
        if not self.playbook:
            return
        sections = list(getattr(self.playbook, "_sections", {}).keys())
        for section in sections:
            ids = list(getattr(self.playbook, "_sections", {}).get(section, []))
            i = 0
            while i < len(ids):
                bullet_i = self.playbook.get_bullet(ids[i])
                if bullet_i is None:
                    i += 1
                    continue
                j = i + 1
                while j < len(ids):
                    bullet_j = self.playbook.get_bullet(ids[j])
                    if bullet_j is None:
                        j += 1
                        continue
                    similarity = SequenceMatcher(None, bullet_i.content or "", bullet_j.content or "").ratio()
                    if similarity >= threshold:
                        new_content = bullet_i.content if len(bullet_i.content or "") >= len(bullet_j.content or "") else bullet_j.content
                        metadata = {
                            "helpful": (getattr(bullet_i, "helpful", 0) or 0) + (getattr(bullet_j, "helpful", 0) or 0),
                            "harmful": (getattr(bullet_i, "harmful", 0) or 0) + (getattr(bullet_j, "harmful", 0) or 0),
                            "neutral": (getattr(bullet_i, "neutral", 0) or 0) + (getattr(bullet_j, "neutral", 0) or 0),
                        }
                        self.playbook.update_bullet(ids[i], content=new_content, metadata=metadata)
                        self.playbook.remove_bullet(ids[j])
                        ids.pop(j)
                        continue
                    j += 1
                i += 1

    def _save_playbook(self) -> None:
        if not self.playbook:
            return
        with self.components.lock:
            self.playbook.save_to_file(str(self.components.playbook_path))

    def _fallback_analysis(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        insights = []
        for dialogue in dialogues:
            messages = dialogue.get("messages", [])
            for idx in range(len(messages) - 1):
                if messages[idx].get("role") == "visitor" and messages[idx + 1].get("role") == "agent":
                    question = messages[idx].get("content", "")
                    answer = messages[idx + 1].get("content", "")
                    if question and answer:
                        analysis = self.analyzer.analyze_dialogue(question, answer)
                        insights.append(
                            {
                                "question": question[:50] + "...",
                                "quality_score": analysis["metrics"].get("overall_score", 0.0),
                                "suggestions": analysis["suggestions"],
                            }
                        )
        return {
            "status": "success",
            "mode": "rule_based_only",
            "dialogues_analyzed": len(insights),
            "insights": insights[:10],
        }

    def get_insights(self) -> Dict[str, Any]:
        if not self.playbook:
            return {"count": 0, "insights": []}
        bullets = self.playbook.bullets()
        return {
            "count": len(bullets),
            "insights": [
                {
                    "content": bullet.content,
                    "helpful": getattr(bullet, "helpful_count", 0),
                    "harmful": getattr(bullet, "harmful_count", 0),
                }
                for bullet in bullets[:20]
            ],
        }


class DifyACEIntegration:
    """Caches webhook messages and triggers learning when ready."""

    def __init__(self, registry: ACEComponentsRegistry):
        self.dialogue_cache: List[Dict[str, Any]] = []
        self.last_learning_time = 0.0
        self.registry = registry
        self.learners: Dict[str, ACELearningManager] = {}
        self._get_learner("customer_service")

    def _get_learner(self, task_type: str) -> ACELearningManager:
        if task_type not in self.learners:
            components = self.registry.get(task_type)
            self.learners[task_type] = ACELearningManager(components)
        return self.learners[task_type]

    def add_dialogue(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.dialogue_cache.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        )
        if len(self.dialogue_cache) > CACHE_SIZE * 3:
            self.dialogue_cache = self.dialogue_cache[-CACHE_SIZE * 2 :]
        logger.debug("Dialogue cached (%s) size=%d", role, len(self.dialogue_cache))

    def should_learn(self) -> bool:
        cache_full = len(self.dialogue_cache) >= CACHE_SIZE
        interval_passed = (time.time() - self.last_learning_time) >= TRAINING_INTERVAL
        return cache_full and interval_passed

    def format_dialogues_for_learning(self) -> List[Dict[str, Any]]:
        sessions: Dict[str, List[Dict[str, Any]]] = {}
        for record in self.dialogue_cache:
            sessions.setdefault(record.get("session_id", "unknown"), []).append(
                {
                    "role": record.get("role"),
                    "content": record.get("content"),
                    "metadata": record.get("metadata", {}),
                }
            )
        return [
            {
                "session_id": sid,
                "messages": msgs,
                "timestamp": datetime.now().isoformat(),
            }
            for sid, msgs in sessions.items()
        ]

    def trigger_learning(self, task_type: str = "customer_service") -> Dict[str, Any]:
        dialogues = self.format_dialogues_for_learning()
        if not dialogues:
            return {"error": "No dialogues to learn from"}
        learner = self._get_learner(task_type)
        result = learner.learn_from_dialogues(dialogues)
        if result.get("status") == "success":
            self.last_learning_time = time.time()
            self.dialogue_cache = []
        return result

    def get_insights(self, task_type: str = "customer_service") -> Dict[str, Any]:
        learner = self._get_learner(task_type)
        return learner.get_insights()


# ---------------------------------------------------------------------------
# Human review pipeline
# ---------------------------------------------------------------------------


class HumanReviewData:
    """Encapsulates one human review submission."""

    def __init__(self, original: str, modified: str, context: List[Dict[str, str]], task_type: str):
        self.original = original
        self.modified = modified
        self.context = context
        self.task_type = task_type
        self.timestamp = datetime.now().isoformat()
        self.feedback = ""
        self.ground_truth = ""
        if "。" in modified:
            parts = modified.split("。", 1)
            self.feedback = parts[0].strip()
            self.ground_truth = parts[1].strip() if len(parts) > 1 else ""
        else:
            self.feedback = modified.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "modified": self.modified,
            "feedback": self.feedback,
            "ground_truth": self.ground_truth,
            "context": self.context,
            "timestamp": self.timestamp,
            "task_type": self.task_type,
        }

    def format_context_as_dialogue(self) -> str:
        if not self.context:
            return ""
        lines = []
        for idx, msg in enumerate(self.context, 1):
            if "agent" in msg:
                lines.append(f"{idx}. [客服] {msg['agent']}")
            elif "visitor" in msg:
                lines.append(f"{idx}. [访客] {msg['visitor']}")
        return "\n".join(lines)

    def extract_question_from_context(self) -> str:
        if not self.context:
            return "(无上下文)"
        for msg in reversed(self.context):
            if "visitor" in msg:
                return msg["visitor"]
        return "(无访客问题)"


class HumanReviewLearner:
    """Runs ACE reflection pipeline on human-reviewed samples."""

    def __init__(self, components: ACEComponents):
        self.components = components
        self.task_type = components.task_type
        self.playbook = components.playbook
        self.reflector = components.reflector
        self.curator = components.curator

    def learn_from_review(self, review_data: HumanReviewData) -> Dict[str, Any]:
        if not self.components.ready or not self.playbook or not self.reflector or not self.curator:
            return {"error": "ACE components not available"}
        try:
            logger.info("👤 Human review learning task (task_type=%s)", self.task_type)
            start = time.time()
            question = self._build_question_context(review_data)
            generator_output = GeneratorOutput(
                reasoning=f"人工审阅: {review_data.feedback}",
                final_answer=review_data.original,
                bullet_ids=[],
                raw={"review_data": review_data.to_dict()},
            )
            reflection = self.reflector.reflect(
                question=question,
                generator_output=generator_output,
                playbook=self.playbook,
                ground_truth=review_data.ground_truth or None,
                feedback=f"Human: {review_data.feedback}",
            )
            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=self.task_type,
                progress="human_review",
            )
            insights_added = 0
            if curator_output and getattr(curator_output, "delta", None):
                with self.components.lock:
                    self.playbook.apply_delta(curator_output.delta)
                insights_added = len([op for op in curator_output.delta.operations if op.type.upper() == "ADD"])
            self._save_review_log(review_data, reflection)
            self._save_playbook()
            elapsed = time.time() - start
            logger.info("✅ Human review learning finished in %.2fs", elapsed)
            return {
                "status": "success",
                "learning_time": elapsed,
                "insights_added": insights_added,
                "total_insights": len(self.playbook.bullets()) if self.playbook else 0,
                "feedback_used": review_data.feedback,
                "ground_truth_used": bool(review_data.ground_truth),
            }
        except Exception as exc:
            logger.exception("❌ Review learning failed")
            return {"error": str(exc)}

    def _build_question_context(self, review_data: HumanReviewData) -> str:
        context_str = review_data.format_context_as_dialogue()
        question = review_data.extract_question_from_context()
        if context_str:
            return (
                f"""[对话历史]
{context_str}

[当前问题]
访客: {question}

[需要改进的回答]
客服（原始）: {review_data.original}
"""
            )
        return (
            f"""[当前问题]
访客: {question}

[需要改进的回答]
客服（原始）: {review_data.original}
"""
        )

    def _save_review_log(self, review_data: HumanReviewData, reflection: Any) -> None:
        try:
            log_entry = {
                "timestamp": review_data.timestamp,
                "task_type": review_data.task_type,
                "original": review_data.original,
                "feedback": review_data.feedback,
                "ground_truth": review_data.ground_truth,
                "context_length": len(review_data.context),
                "reflection": {
                    "key_insight": getattr(reflection, "key_insight", "")[:500],
                    "error_identification": getattr(reflection, "error_identification", "")[:500],
                },
            }
            log_file = REVIEW_LOG_DIR / f"reviews_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write review log: %s", exc)

    def _save_playbook(self) -> None:
        if not self.playbook:
            return
        with self.components.lock:
            self.playbook.save_to_file(str(self.components.playbook_path))


class HumanReviewService:
    """HTTP-friendly wrapper adding storage + async execution."""

    def __init__(self, registry: ACEComponentsRegistry):
        self.registry = registry
        self.learners: Dict[str, HumanReviewLearner] = {}
        self.storage_lock = threading.Lock()

    def _get_learner(self, task_type: str) -> HumanReviewLearner:
        if task_type not in self.learners:
            self.learners[task_type] = HumanReviewLearner(self.registry.get(task_type))
        return self.learners[task_type]

    def load_review_data(self) -> List[Dict[str, Any]]:
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.error("Error loading review data: %s", exc)
            return []

    def save_review_data(self, data: List[Dict[str, Any]]) -> bool:
        try:
            with open(DATA_FILE, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            logger.error("Error saving review data: %s", exc)
            return False

    @staticmethod
    def _validate_payload(payload: Dict[str, Any]) -> Optional[str]:
        if not payload:
            return "No data received"
        missing = [field for field in ("original", "modified") if not payload.get(field)]
        if missing:
            return f"Missing required fields: {', '.join(missing)}"
        return None

    def _process_payload(self, payload: Dict[str, Any]) -> Tuple[Optional[HumanReviewData], Optional[Dict[str, Any]], Optional[str]]:
        error = self._validate_payload(payload)
        if error:
            return None, None, error
        task_type = payload.get("task_type", "customer_service")
        review = HumanReviewData(
            original=payload["original"],
            modified=payload["modified"],
            context=payload.get("context", []),
            task_type=task_type,
        )
        record = review.to_dict()
        for key, value in payload.items():
            if key not in record:
                record[key] = value
        return review, record, None

    def append_record(self, record: Dict[str, Any]) -> int:
        with self.storage_lock:
            reviews = self.load_review_data()
            reviews.append(record)
            if not self.save_review_data(reviews):
                raise IOError("Failed to persist review data")
            return len(reviews)

    def trigger_learning_async(self, review: HumanReviewData) -> None:
        learner = self._get_learner(review.task_type)

        def _run() -> None:
            try:
                learner.learn_from_review(review)
            except Exception:
                logger.exception("Background human review learning failed")

        threading.Thread(target=_run, daemon=True).start()

    def batch_learn(self, reviews: List[HumanReviewData]) -> Dict[str, Any]:
        results = []
        total_added = 0
        for review in reviews:
            result = self._get_learner(review.task_type).learn_from_review(review)
            results.append(result)
            if result.get("status") == "success":
                total_added += result.get("insights_added", 0)
        return {
            "status": "success",
            "total_reviews": len(reviews),
            "successful": sum(1 for res in results if res.get("status") == "success"),
            "total_insights_added": total_added,
            "results": results,
        }


# ---------------------------------------------------------------------------
# Flask application setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)
ace_registry = ACEComponentsRegistry()
dify_integration = DifyACEIntegration(ace_registry)
review_service = HumanReviewService(ace_registry)
stop_event = threading.Event()


def background_learning_loop(interval: int = BACKGROUND_POLL_INTERVAL) -> None:
    logger.info("🔄 Background learner started (interval=%ss)", interval)
    while not stop_event.is_set():
        try:
            if dify_integration.should_learn():
                logger.info("🤖 Background trigger fired")
                result = dify_integration.trigger_learning()
                logger.info("Background learning status: %s", result.get("status"))
        except Exception as exc:
            logger.error("Background loop error: %s", exc)
        stop_event.wait(interval)


def _start_background_thread() -> None:
    threading.Thread(target=background_learning_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Auto-learning routes
# ---------------------------------------------------------------------------


@app.route("/webhook", methods=["POST"])
def webhook() -> Any:
    try:
        data = request.json or {}
        session_id = data.get("conversation_id") or data.get("session_id") or "unknown"
        system_prompt = data.get("prompt") or data.get("system_prompt")
        user_satisfaction = None
        if isinstance(data.get("feedback"), dict):
            user_satisfaction = data["feedback"].get("rating")
        if "query" in data:
            dify_integration.add_dialogue(
                session_id,
                "visitor",
                data.get("query"),
                {"user_satisfaction": user_satisfaction, "system_prompt": system_prompt},
            )
        if "answer" in data:
            dify_integration.add_dialogue(session_id, "agent", data.get("answer"))
        return jsonify(
            {
                "status": "ok",
                "cached": len(dify_integration.dialogue_cache),
                "next_learning_in": max(0, TRAINING_INTERVAL - (time.time() - dify_integration.last_learning_time)),
            }
        )
    except Exception as exc:
        logger.exception("Webhook error")
        return jsonify({"error": str(exc)}), 500


@app.route("/manual_learn", methods=["POST"])
def manual_learn() -> Any:
    try:
        task_type = (request.json or {}).get("task_type", "customer_service")
        result = dify_integration.trigger_learning(task_type=task_type)
        insights = dify_integration.get_insights(task_type=task_type)
        return jsonify({"learning_result": result, "insights": insights})
    except Exception as exc:
        logger.exception("Manual learn error")
        return jsonify({"error": str(exc)}), 500


@app.route("/insights/<task_type>", methods=["GET"])
def get_insights(task_type: str) -> Any:
    try:
        return jsonify(dify_integration.get_insights(task_type))
    except Exception as exc:
        logger.exception("Insights error")
        return jsonify({"error": str(exc)}), 500


@app.route("/status", methods=["GET"])
def status() -> Any:
    return jsonify(
        {
            "status": "running",
            "mode": "learning_only",
            "ace_available": ACE_AVAILABLE,
            "model": ACE_MODEL,
            "dialogue_cache_size": len(dify_integration.dialogue_cache),
            "cache_threshold": CACHE_SIZE,
            "last_learning": dify_integration.last_learning_time,
            "next_learning_in": max(0, TRAINING_INTERVAL - (time.time() - dify_integration.last_learning_time)),
        }
    )


# ---------------------------------------------------------------------------
# Human review routes
# ---------------------------------------------------------------------------


@app.route("/submit_review", methods=["POST"])
def submit_review() -> Any:
    try:
        payload = request.json or {}
        review, record, error = review_service._process_payload(payload)
        if error:
            return jsonify({"error": error}), 400
        try:
            total_reviews = review_service.append_record(record)
        except IOError:
            logger.exception("Failed to persist review data")
            return jsonify({"error": "Failed to save review data"}), 500
        review_service.trigger_learning_async(review)
        return jsonify(
            {
                "success": True,
                "message": "Review stored and learning queued",
                "total_reviews": total_reviews,
                "timestamp": record.get("timestamp"),
            }
        )
    except Exception as exc:
        logger.exception("Submit review error")
        return jsonify({"error": str(exc)}), 500


@app.route("/batch_submit_reviews", methods=["POST"])
def batch_submit_reviews() -> Any:
    try:
        data = request.json or {}
        payloads = data.get("reviews", [])
        if not payloads:
            return jsonify({"error": "No reviews provided"}), 400
        reviews: List[HumanReviewData] = []
        records: List[Dict[str, Any]] = []
        for idx, payload in enumerate(payloads):
            review, record, error = review_service._process_payload(payload)
            if error:
                return jsonify({"error": f"Invalid review at index {idx}: {error}"}), 400
            reviews.append(review)
            records.append(record)
        stored = review_service.load_review_data()
        stored.extend(records)
        if not review_service.save_review_data(stored):
            return jsonify({"error": "Failed to save review data"}), 500
        summary = review_service.batch_learn(reviews)
        return jsonify(summary)
    except Exception as exc:
        logger.exception("Batch submit error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/review", methods=["POST"])
def api_receive_review() -> Any:
    try:
        payload = request.json or {}
        review, record, error = review_service._process_payload(payload)
        if error:
            return jsonify({"success": False, "error": error}), 400
        try:
            total_reviews = review_service.append_record(record)
        except IOError:
            logger.exception("Failed to persist review record via /api/review")
            return jsonify({"success": False, "error": "Failed to save data"}), 500
        review_service.trigger_learning_async(review)
        return jsonify(
            {
                "success": True,
                "data": record,
                "total_reviews": total_reviews,
                "learning": {"status": "queued", "message": "Learning task scheduled in background"},
            }
        )
    except Exception as exc:
        logger.exception("Error processing /api/review request")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/reviews", methods=["GET"])
def get_reviews() -> Any:
    try:
        return jsonify({"success": True, "data": review_service.load_review_data()})
    except Exception as exc:
        logger.exception("Error retrieving review list")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/analytics", methods=["GET"])
def get_analytics() -> Any:
    try:
        reviews = review_service.load_review_data()
        modifications = []
        total_length_diff = 0
        total_original_length = 0
        total_modified_length = 0
        for record in reviews:
            original = record.get("original", "")
            modified = record.get("modified", "")
            if original and modified:
                original_len = len(original)
                modified_len = len(modified)
                length_diff = modified_len - original_len
                modifications.append(
                    {
                        "original_length": original_len,
                        "modified_length": modified_len,
                        "length_diff": length_diff,
                    }
                )
                total_length_diff += length_diff
                total_original_length += original_len
                total_modified_length += modified_len
        avg_length_diff = total_length_diff / len(modifications) if modifications else 0
        avg_original_length = total_original_length / len(modifications) if modifications else 0
        avg_modified_length = total_modified_length / len(modifications) if modifications else 0
        similarity_scores = []
        for record in reviews:
            original = record.get("original", "")
            modified = record.get("modified", "")
            if original and modified:
                similarity_scores.append(SequenceMatcher(None, original, modified).ratio())
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        analytics = {
            "total_reviews": len(reviews),
            "average_length_difference": round(avg_length_diff, 2),
            "average_original_length": round(avg_original_length, 2),
            "average_modified_length": round(avg_modified_length, 2),
            "average_text_similarity": round(avg_similarity, 4),
            "modifications": modifications[:10],
        }
        return jsonify({"success": True, "data": analytics})
    except Exception as exc:
        logger.exception("Error performing analytics")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/review_stats", methods=["GET"])
def review_stats() -> Any:
    try:
        today = datetime.now().strftime("%Y%m%d")
        log_file = REVIEW_LOG_DIR / f"reviews_{today}.jsonl"
        review_count = 0
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as handle:
                review_count = sum(1 for _ in handle)
        learner = review_service._get_learner("customer_service")
        return jsonify(
            {
                "status": "running",
                "playbook_path": str(learner.components.playbook_path),
                "total_insights": len(learner.playbook.bullets()) if learner.playbook else 0,
                "today_reviews": review_count,
                "ace_available": ACE_AVAILABLE,
            }
        )
    except Exception as exc:
        logger.exception("Stats error")
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health() -> Any:
    learner = review_service._get_learner("customer_service")
    return jsonify(
        {
            "status": "healthy",
            "service": "combined_learning",
            "ace_available": ACE_AVAILABLE,
            "reflector_ready": learner.reflector is not None,
            "curator_ready": learner.curator is not None,
        }
    )


@app.route("/", methods=["GET"])
def home() -> Any:
    return jsonify(
        {
            "message": "Combined Dify & Human Review Learning Service",
            "endpoints": {
                "POST /webhook": "缓存 Dify 对话并学习",
                "POST /manual_learn": "手动触发学习",
                "GET /insights/<task>": "查看洞察",
                "POST /submit_review": "提交人工审阅",
                "POST /batch_submit_reviews": "批量人工审阅",
                "GET /api/analytics": "审阅分析",
                "GET /status": "服务状态",
                "GET /health": "健康检查",
            },
        }
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 Combined Learning Service")
    print("=" * 60)
    print(f"ACE Available: {ACE_AVAILABLE}")
    print(f"Model: {ACE_MODEL}")
    print(f"Playbooks: {PLAYBOOK_DIR}")
    print(f"Reviews: {REVIEW_LOG_DIR}")
    print("=" * 60 + "\n")
    _start_background_thread()
    app.run(host="0.0.0.0", port=18578, debug=False)
