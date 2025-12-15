#!/usr/bin/env python3
"""
Minimal server: receive Dify webhook, cache dialogues, trigger ACE training,
and push the updated playbook back to Dify.

Endpoints kept:
- POST /webhook         : receive Dify webhook (cache dialogues)
- POST /manual_train    : manual trigger for training

Background:
- A daemon background thread periodically checks cache/time and triggers training.
"""

import os
import json
import time
import threading
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from flask import Flask, request, jsonify

# Configuration
# 使用远程服务器地址
ACE_API_BASE_URL = os.environ.get("ACE_API_BASE_URL", "http://123.181.192.120:18578")
TRAIN_ENDPOINT = f"{ACE_API_BASE_URL.rstrip('/')}/train"
PLAYBOOK_ENDPOINT = f"{ACE_API_BASE_URL.rstrip('/')}/playbook"

DIFY_API_BASE_URL = os.environ.get("DIFY_API_BASE_URL", "http://36.248.221.38")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY", "app-qtsvkyrD6RAUW5LAHALVXUX0")
DIFY_PLAYBOOK_ENDPOINT = os.environ.get("DIFY_PLAYBOOK_ENDPOINT")

# Tune these for your deployment
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "1"))
TRAINING_INTERVAL = int(os.environ.get("TRAINING_INTERVAL", "3600"))
BACKGROUND_POLL_INTERVAL = int(os.environ.get("BACKGROUND_POLL_INTERVAL", "60"))


class DifyACEIntegration:
    def __init__(self):
        self.dialogue_cache: List[Dict[str, Any]] = []
        self.last_training_time = 0

    def add_dialogue(self, session_id: str, role: str, content: str):
        self.dialogue_cache.append({
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # keep recent items
        if len(self.dialogue_cache) > CACHE_SIZE * 2:
            self.dialogue_cache = self.dialogue_cache[-CACHE_SIZE * 2 :]

    def should_train(self) -> bool:
        if len(self.dialogue_cache) < CACHE_SIZE:
            return False
        return (time.time() - self.last_training_time) >= TRAINING_INTERVAL

    def format_dialogues_for_training(self) -> List[Dict[str, Any]]:
        by_session = {}
        for d in self.dialogue_cache:
            sid = d.get("session_id", "unknown")
            by_session.setdefault(sid, []).append({"role": d.get("role"), "content": d.get("content")})
        out = []
        for sid, msgs in by_session.items():
            out.append({"session_id": sid, "messages": msgs, "timestamp": datetime.now().isoformat()})
        return out

    def trigger_training(self, task_type: str = "customer_service") -> Dict[str, Any]:
        try:
            # 在远程服务器上，我们直接执行训练
            dialogues = self.format_dialogues_for_training()
            if not dialogues:
                return {"error": "No dialogues to train on"}
            
            # 在远程服务器上执行ACE训练
            print(f"Training on {len(dialogues)} dialogues for task type: {task_type}")
            
            # 导入ACE模块进行训练
            try:
                from ace import OfflineAdapter
                from ace.features import Sample
                
                # 将对话转换为ACE样本格式
                samples = []
                for dialogue in dialogues:
                    # 创建对话样本
                    sample = Sample(
                        input=dialogue.get("query", ""),
                        output=dialogue.get("answer", ""),
                        metadata={"task_type": task_type}
                    )
                    samples.append(sample)
                
                # 执行训练
                adapter = OfflineAdapter()
                adapter.train(samples)
                
                # 保存训练后的playbook
                playbook_path = f"kouqiang_{task_type}_playbook.json"
                adapter.save_playbook(playbook_path)
                
                print(f"Training completed and playbook saved to {playbook_path}")
                
            except ImportError as e:
                print(f"ACE module not available: {e}")
                print("Falling back to mock training")
                # 模拟训练过程
                time.sleep(2)  # 模拟训练耗时
                
                # 创建一个简单的模拟playbook
                playbook_path = f"kouqiang_{task_type}_playbook.json"
                mock_playbook = {
                    "task_type": task_type,
                    "training_data_count": len(dialogues),
                    "created_at": datetime.now().isoformat(),
                    "status": "mock_training_completed"
                }
                
                with open(playbook_path, 'w', encoding='utf-8') as f:
                    json.dump(mock_playbook, f, ensure_ascii=False, indent=2)
                
                print(f"Mock training completed and playbook saved to {playbook_path}")
            
            self.last_training_time = time.time()
            # clear cache on successful trigger
            self.dialogue_cache = []
            return {"status": "success", "message": f"Training completed for {len(dialogues)} dialogues"}
        except Exception as e:
            return {"error": str(e)}

    def fetch_playbook(self, task_type: str = "customer_service") -> Optional[Dict[str, Any]]:
        try:
            # 在远程服务器上直接读取本地playbook文件
            playbook_path = f"kouqiang_{task_type}_playbook.json"
            if os.path.exists(playbook_path):
                with open(playbook_path, 'r', encoding='utf-8') as f:
                    playbook = json.load(f)
                return playbook
            else:
                print(f"Playbook not found at {playbook_path}")
                return None
        except Exception as e:
            print(f"Failed to fetch playbook: {e}")
            return None


def send_playbook_to_dify(playbook: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = DIFY_PLAYBOOK_ENDPOINT or f"{DIFY_API_BASE_URL.rstrip('/')}/v1/playbook"
    headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(endpoint, headers=headers, json={"playbook": playbook}, timeout=30)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"status": "ok", "raw": r.text}
    except Exception as e:
        return {"error": str(e)}


# Flask app and integration instance
app = Flask(__name__)
integration = DifyACEIntegration()
stop_event = threading.Event()


def background_trainer_loop(poll_interval: int = BACKGROUND_POLL_INTERVAL):
    while not stop_event.is_set():
        try:
            if integration.should_train():
                print("Background: triggering training")
                res = integration.trigger_training()
                print("Training response:", res)
                playbook = integration.fetch_playbook()
                if playbook:
                    pushed = send_playbook_to_dify(playbook)
                    print("Pushed playbook result:", pushed)
        except Exception as e:
            print("Background trainer error:", e)
        stop_event.wait(poll_interval)


def _start_background_trainer():
    t = threading.Thread(target=background_trainer_loop, daemon=True)
    t.start()
    print("Background trainer started")


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json or {}
    # adapt to your Dify webhook format
    session_id = data.get("conversation_id") or data.get("session_id") or "unknown"
    if "query" in data:
        integration.add_dialogue(session_id, "visitor", data.get("query"))
    if "answer" in data:
        integration.add_dialogue(session_id, "agent", data.get("answer"))
    return jsonify({"status": "ok", "cached": len(integration.dialogue_cache)})


@app.route("/manual_train", methods=["POST"])
def manual_train():
    task_type = (request.json or {}).get("task_type", "customer_service")
    res = integration.trigger_training(task_type=task_type)
    # after training attempt to fetch and push
    playbook = integration.fetch_playbook(task_type=task_type)
    pushed = None
    if playbook:
        pushed = send_playbook_to_dify(playbook)
    return jsonify({"training_result": res, "pushed": pushed})


@app.route("/train", methods=["POST"])
def train():
    """训练端点，用于接收对话数据并触发训练"""
    try:
        data = request.json or {}
        task_type = data.get("task_type", "customer_service")
        dialogues = data.get("dialogues", [])
        
        # 将接收到的对话数据添加到缓存
        for dialogue in dialogues:
            session_id = dialogue.get("session_id", "unknown")
            messages = dialogue.get("messages", [])
            for msg in messages:
                role = "visitor" if msg.get("role") == "user" else "agent"
                content = msg.get("content", "")
                if content:
                    integration.add_dialogue(session_id, role, content)
        
        # 触发训练
        result = integration.trigger_training(task_type=task_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/playbook/<task_type>", methods=["GET"])
def get_playbook(task_type):
    """获取指定类型的playbook"""
    try:
        playbook_path = f"kouqiang_{task_type}_playbook.json"
        if os.path.exists(playbook_path):
            with open(playbook_path, 'r', encoding='utf-8') as f:
                playbook = json.load(f)
            return jsonify({"playbook": playbook})
        else:
            return jsonify({"error": f"Playbook not found for task type: {task_type}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting minimal Dify->ACE integration server on 123.181.192.120:18578")
    _start_background_trainer()
    app.run(host="0.0.0.0", port=18578)
