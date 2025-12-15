#!/usr/bin/env python3
"""
Human Review Learning Module
处理人工审阅数据并传递给 ACE Reflector 进行学习
与自动学习流程分离，避免冲突
"""

import os
import json
import time
import difflib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import logging
import threading

# ACE Framework
try:
    from ace import (
        Reflector,
        Curator,
        Playbook,
        LiteLLMClient,
        GeneratorOutput,
    )
    ACE_AVAILABLE = True
except ImportError:
    print("⚠️  ACE framework not installed. Run: pip install ace-framework")
    ACE_AVAILABLE = False

# Configuration
ACE_MODEL = os.environ.get("ACE_MODEL", "openai/qwen-max-latest")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-25587b057d5242428bb940d44035b5fd")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Storage
PLAYBOOK_DIR = Path("/data/fangsy/jhq/kouqiangACE/agentic-context-engine-main/dify_realtime/playbooks")
PLAYBOOK_DIR.mkdir(exist_ok=True)
REVIEW_LOG_DIR = Path("/data/fangsy/jhq/kouqiangACE/agentic-context-engine-main/dify_realtime/reviews")
REVIEW_LOG_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# File handler for review logs
review_log_file = REVIEW_LOG_DIR / "human_review.log"
fh = logging.FileHandler(str(review_log_file), encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(fh)


DATA_FILE = Path(os.environ.get("REVIEW_DATA_FILE", "review_data.json")).resolve()
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
if not DATA_FILE.exists():
    DATA_FILE.write_text("[]", encoding='utf-8')


def load_review_data() -> List[Dict[str, Any]]:
    """加载审阅数据文件"""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Error loading review data: %s", exc)
        return []


def save_review_data(data: List[Dict[str, Any]]) -> bool:
    """保存审阅数据文件"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as exc:
        logger.error("Error saving review data: %s", exc)
        return False


def _validate_review_payload(payload: Dict[str, Any]) -> Optional[str]:
    """校验审阅请求体"""
    if not payload:
        return "No data received"
    missing = [field for field in ("original", "modified") if not payload.get(field)]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    return None


def _merge_additional_fields(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """补充额外字段到存储记录"""
    for key, value in source.items():
        if key not in target:
            target[key] = value


def _append_review_record(record: Dict[str, Any]) -> int:
    """追加一条审阅记录并返回总数"""
    reviews = load_review_data()
    reviews.append(record)
    if not save_review_data(reviews):
        raise IOError("Failed to persist review data")
    return len(reviews)


class HumanReviewData:
    """人工审阅数据模型"""
    
    def __init__(self, original: str, modified: str, context: List[Dict[str, str]]):
        self.original = original
        self.modified = modified
        self.context = context
        self.timestamp = datetime.now().isoformat()
        
        # 解析 modified: "。"前为 feedback，"。"后为 ground_truth
        self.feedback = ""
        self.ground_truth = ""
        
        if "。" in modified:
            parts = modified.split("。", 1)
            self.feedback = parts[0].strip()
            self.ground_truth = parts[1].strip() if len(parts) > 1 else ""
        else:
            # 如果没有"。"，全部作为 feedback
            self.feedback = modified.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "original": self.original,
            "modified": self.modified,
            "feedback": self.feedback,
            "ground_truth": self.ground_truth,
            "context": self.context,
            "timestamp": self.timestamp
        }
    
    def format_context_as_dialogue(self) -> str:
        """将 context 格式化为对话历史字符串"""
        if not self.context:
            return ""
        
        dialogue_lines = []
        for idx, msg in enumerate(self.context, 1):
            role = msg.get("agent", msg.get("visitor", ""))
            speaker = "客服" if "agent" in msg else "访客"
            if role:
                dialogue_lines.append(f"{idx}. [{speaker}] {role}")
        
        return "\n".join(dialogue_lines)
    
    def extract_question_from_context(self) -> str:
        """从 context 中提取最后一个访客问题"""
        if not self.context:
            return "(无上下文)"
        
        # 倒序查找最后一个访客消息
        for msg in reversed(self.context):
            if "visitor" in msg:
                return msg["visitor"]
        
        return "(无访客问题)"


class HumanReviewLearner:
    """人工审阅学习器"""
    
    def __init__(self, task_type: str = "customer_service"):
        self.task_type = task_type
        self.playbook_path = PLAYBOOK_DIR / f"{task_type}_insights.json"
        
        if not ACE_AVAILABLE or not OPENAI_API_KEY:
            print("⚠️  ACE not available, review learning disabled")
            self.reflector = None
            self.curator = None
            self.playbook = None
            return
        
        try:
            self.llm = LiteLLMClient(
                model=ACE_MODEL,
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_API_BASE,
                temperature=0.3,
                max_tokens=4000,
                timeout=60
            )
            
            # 加载或创建 Playbook
            if self.playbook_path.exists():
                logger.info(f"📚 Loading playbook from {self.playbook_path}")
                self.playbook = Playbook.load_from_file(str(self.playbook_path))
            else:
                logger.info(f"📚 Creating new playbook")
                self.playbook = Playbook()
            
            self.reflector = Reflector(self.llm)
            self.curator = Curator(self.llm)
            
            logger.info(f"✅ Human Review Learner initialized")
            logger.info(f"   Playbook insights: {len(self.playbook.bullets())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            self.reflector = None
            self.curator = None
    
    def learn_from_review(self, review_data: HumanReviewData) -> Dict[str, Any]:
        """
        从人工审阅数据中学习
        不进行自动评分，直接使用人工反馈
        """
        if not self.reflector or not self.curator:
            return {"error": "ACE components not available"}
        
        try:
            logger.info("="*60)
            logger.info("👤 Human Review Learning")
            logger.info("="*60)
            logger.info(f"Original: {review_data.original}")
            logger.info(f"Feedback: {review_data.feedback}")
            logger.info(f"Ground Truth: {review_data.ground_truth}")
            logger.info(f"Context Length: {len(review_data.context)}")
            
            start_time = time.time()
            
            # 1. 构建问题上下文
            question = self._build_question_context(review_data)
            
            # 2. 构建 GeneratorOutput（原始回答）
            gen_output = GeneratorOutput(
                reasoning=f"人工审阅: {review_data.feedback}",
                final_answer=review_data.original,
                bullet_ids=[],
                raw={"review_data": review_data.to_dict()}
            )
            
            # 3. 调用 Reflector 进行反思
            # ground_truth 参数接收改进后的标准答案
            reflector_output = self.reflector.reflect(
                question=question,
                generator_output=gen_output,
                playbook=self.playbook,
                ground_truth=review_data.ground_truth if review_data.ground_truth else None,
                feedback=f"Human: {review_data.feedback}"
            )
            
            logger.info(f"反思完成: {getattr(reflector_output, 'key_insight', 'N/A')[:200]}")
            
            # 4. 使用 Curator 更新 Playbook
            curator_output = self.curator.curate(
                reflection=reflector_output,
                playbook=self.playbook,
                question_context=self.task_type,
                progress="human_review"
            )
            
            # 5. 应用 delta
            insights_added = 0
            if curator_output and getattr(curator_output, 'delta', None):
                delta = curator_output.delta
                ops = getattr(delta, 'operations', []) or []
                
                logger.info(f"Curator delta: {len(ops)} operations")
                
                self.playbook.apply_delta(delta)
                insights_added = len([o for o in ops if o.type.upper() == 'ADD'])
            
            # 6. 保存 Playbook
            self.playbook.save_to_file(str(self.playbook_path))
            
            # 7. 记录审阅日志
            self._save_review_log(review_data, reflector_output)
            
            learning_time = time.time() - start_time
            
            logger.info("="*60)
            logger.info(f"✅ Review Learning Completed in {learning_time:.2f}s")
            logger.info(f"   Insights added: {insights_added}")
            logger.info(f"   Total insights: {len(self.playbook.bullets())}")
            logger.info("="*60)
            
            return {
                "status": "success",
                "learning_time": learning_time,
                "insights_added": insights_added,
                "total_insights": len(self.playbook.bullets()),
                "feedback_used": review_data.feedback,
                "ground_truth_used": bool(review_data.ground_truth)
            }
            
        except Exception as e:
            logger.exception("❌ Review learning failed")
            return {"error": str(e)}
    
    def _build_question_context(self, review_data: HumanReviewData) -> str:
        """构建包含上下文的问题"""
        context_str = review_data.format_context_as_dialogue()
        question = review_data.extract_question_from_context()
        
        if context_str:
            return f"""[对话历史]
{context_str}

[当前问题]
访客: {question}

[需要改进的回答]
客服（原始）: {review_data.original}
"""
        else:
            return f"""[当前问题]
访客: {question}

[需要改进的回答]
客服（原始）: {review_data.original}
"""
    
    def _save_review_log(self, review_data: HumanReviewData, reflector_output: Any):
        """保存审阅日志到文件"""
        try:
            log_entry = {
                "timestamp": review_data.timestamp,
                "original": review_data.original,
                "feedback": review_data.feedback,
                "ground_truth": review_data.ground_truth,
                "context_length": len(review_data.context),
                "reflection": {
                    "key_insight": getattr(reflector_output, 'key_insight', '')[:500],
                    "error_identification": getattr(reflector_output, 'error_identification', '')[:500],
                }
            }
            
            log_file = REVIEW_LOG_DIR / f"reviews_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.warning(f"Failed to save review log: {e}")
    
    def batch_learn_from_reviews(self, reviews: List[HumanReviewData]) -> Dict[str, Any]:
        """批量学习人工审阅数据"""
        results = []
        total_insights_added = 0
        
        logger.info(f"📚 Batch learning from {len(reviews)} reviews")
        
        for idx, review in enumerate(reviews, 1):
            logger.info(f"Processing review {idx}/{len(reviews)}")
            result = self.learn_from_review(review)
            results.append(result)
            
            if result.get("status") == "success":
                total_insights_added += result.get("insights_added", 0)
        
        return {
            "status": "success",
            "total_reviews": len(reviews),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "total_insights_added": total_insights_added,
            "results": results
        }


# Flask App for Human Review Service
app = Flask(__name__)
CORS(app)
learner = HumanReviewLearner()


def _trigger_learning_async(review_data: HumanReviewData) -> None:
    """在后台线程中触发学习，避免阻塞 HTTP 响应"""
    if not learner:
        return

    def _run():
        try:
            learner.learn_from_review(review_data)
        except Exception:
            logger.exception("Learning task failed")

    threading.Thread(target=_run, daemon=True).start()


def _process_review_payload(payload: Dict[str, Any]) -> Tuple[Optional[HumanReviewData], Optional[Dict[str, Any]], Optional[str]]:
    """构建审阅对象和存储记录"""
    error = _validate_review_payload(payload)
    if error:
        return None, None, error
    review_data = HumanReviewData(
        original=payload["original"],
        modified=payload["modified"],
        context=payload.get("context", [])
    )
    record = review_data.to_dict()
    _merge_additional_fields(record, payload)
    return review_data, record, None


@app.route("/submit_review", methods=["POST"])
def submit_review():
    """提交单条审阅并触发学习"""
    try:
        payload = request.json or {}
        review_data, record, error = _process_review_payload(payload)
        if error:
            return jsonify({"error": error}), 400
        try:
            total_reviews = _append_review_record(record)
        except IOError:
            logger.exception("❌ Failed to persist review record")
            return jsonify({"error": "Failed to save review data"}), 500
        logger.info("📨 Received review submission")
        logger.info(f"   Feedback: {review_data.feedback}")
        logger.info(f"   Ground Truth: {review_data.ground_truth}")
        _trigger_learning_async(review_data)
        return jsonify({
            "success": True,
            "message": "Review stored and learning queued",
            "total_reviews": total_reviews,
            "timestamp": record.get("timestamp")
        })
    except Exception as e:
        logger.exception("❌ Submit review error")
        return jsonify({"error": str(e)}), 500


@app.route("/batch_submit_reviews", methods=["POST"])
def batch_submit_reviews():
    """批量提交人工审阅数据并触发学习"""
    try:
        data = request.json or {}
        reviews_data = data.get("reviews", [])
        if not reviews_data:
            return jsonify({"error": "No reviews provided"}), 400

        prepared: List[Tuple[HumanReviewData, Dict[str, Any]]] = []
        for idx, review_payload in enumerate(reviews_data):
            review_data, record, error = _process_review_payload(review_payload)
            if error:
                return jsonify({"error": f"Invalid review at index {idx}: {error}"}), 400
            prepared.append((review_data, record))

        logger.info(f"📨 Received batch submission: {len(prepared)} reviews")

        stored_reviews = load_review_data()
        results = []
        for review_data, record in prepared:
            stored_reviews.append(record)
            results.append(learner.learn_from_review(review_data))

        if not save_review_data(stored_reviews):
            return jsonify({"error": "Failed to save review data"}), 500

        summary = {
            "status": "success",
            "total_reviews": len(prepared),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "results": results
        }
        return jsonify(summary)
    except Exception as e:
        logger.exception("❌ Batch submit error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/review', methods=['POST'])
def api_receive_review():
    """接收审阅数据并写入本地文件与学习流程"""
    try:
        payload = request.get_json() or {}
        review_data, record, error = _process_review_payload(payload)
        if error:
            return jsonify({'success': False, 'error': error}), 400
        try:
            total_reviews = _append_review_record(record)
        except IOError:
            logger.exception("❌ Failed to persist review record via /api/review")
            return jsonify({'success': False, 'error': 'Failed to save data'}), 500
        _trigger_learning_async(review_data)
        return jsonify({
            'success': True,
            'data': record,
            'total_reviews': total_reviews,
            'learning': {
                'status': 'queued',
                'message': 'Learning task scheduled in background'
            }
        }), 200
    except Exception as e:
        logger.exception("Error processing /api/review request")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reviews', methods=['GET'])
def get_reviews():
    """获取所有审阅数据"""
    try:
        return jsonify({'success': True, 'data': load_review_data()}), 200
    except Exception as e:
        logger.exception("Error retrieving review list")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """获取数据分析结果"""
    try:
        reviews = load_review_data()
        total_reviews = len(reviews)
        modifications = []
        total_length_diff = 0
        total_original_length = 0
        total_modified_length = 0

        for review in reviews:
            original = review.get('original', '')
            modified = review.get('modified', '')
            if original and modified:
                original_len = len(original)
                modified_len = len(modified)
                length_diff = modified_len - original_len
                modifications.append({
                    'original_length': original_len,
                    'modified_length': modified_len,
                    'length_diff': length_diff
                })
                total_length_diff += length_diff
                total_original_length += original_len
                total_modified_length += modified_len

        avg_length_diff = total_length_diff / len(modifications) if modifications else 0
        avg_original_length = total_original_length / len(modifications) if modifications else 0
        avg_modified_length = total_modified_length / len(modifications) if modifications else 0

        similarity_scores = []
        for review in reviews:
            original = review.get('original', '')
            modified = review.get('modified', '')
            if original and modified:
                similarity = difflib.SequenceMatcher(None, original, modified).ratio()
                similarity_scores.append(similarity)

        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

        analytics = {
            'total_reviews': total_reviews,
            'average_length_difference': round(avg_length_diff, 2),
            'average_original_length': round(avg_original_length, 2),
            'average_modified_length': round(avg_modified_length, 2),
            'average_text_similarity': round(avg_similarity, 4),
            'modifications': modifications[:10]
        }

        return jsonify({'success': True, 'data': analytics}), 200
    except Exception as e:
        logger.exception("Error performing analytics")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """根路径健康说明"""
    return jsonify({
        'message': 'Human Review Learning Server is running',
        'endpoints': {
            'POST /api/review': '接收审阅数据并触发学习',
            'GET /api/reviews': '获取所有审阅数据',
            'GET /api/analytics': '获取审阅分析结果',
            'POST /submit_review': '兼容的人工审阅提交入口',
            'POST /batch_submit_reviews': '批量人工审阅提交入口',
            'GET /review_stats': '学习统计',
            'GET /health': '健康检查'
        }
    }), 200


@app.route("/review_stats", methods=["GET"])
def review_stats():
    """获取审阅学习统计"""
    try:
        # 统计今日审阅日志
        today = datetime.now().strftime('%Y%m%d')
        log_file = REVIEW_LOG_DIR / f"reviews_{today}.jsonl"
        
        review_count = 0
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                review_count = sum(1 for _ in f)
        
        return jsonify({
            "status": "running",
            "playbook_path": str(learner.playbook_path),
            "total_insights": len(learner.playbook.bullets()) if learner.playbook else 0,
            "today_reviews": review_count,
            "ace_available": ACE_AVAILABLE
        })
        
    except Exception as e:
        logger.exception("❌ Stats error")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "human_review_learner",
        "ace_available": ACE_AVAILABLE,
        "reflector_ready": learner.reflector is not None,
        "curator_ready": learner.curator is not None
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("👤 Human Review Learning Service")
    print("="*60)
    print(f"Mode: Human Review Only")
    print(f"ACE Available: {ACE_AVAILABLE}")
    print(f"Model: {ACE_MODEL}")
    print(f"Playbook: {learner.playbook_path}")
    print(f"Review Logs: {REVIEW_LOG_DIR}")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /                   - 服务状态与端点列表")
    print("  POST /api/review         - 接收并学习单条审阅")
    print("  GET  /api/reviews        - 查看所有审阅记录")
    print("  GET  /api/analytics      - 查看审阅分析")
    print("  POST /submit_review        - 提交单条审阅")
    print("  POST /batch_submit_reviews - 批量提交审阅")
    print("  GET  /review_stats         - 审阅统计")
    print("  GET  /health               - 健康检查")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=18580, debug=False)
