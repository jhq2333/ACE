#!/usr/bin/env python3
"""
Dify机器人与ACE训练API集成示例

展示如何将Dify机器人的对话数据发送到ACE训练API，
并获取更新后的playbook用于改进机器人回复。
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# ==================== 配置 ====================
# ACE训练API配置
ACE_API_BASE_URL = "http://localhost:8000"
TRAIN_ENDPOINT = f"{ACE_API_BASE_URL}/train"
PLAYBOOK_ENDPOINT = f"{ACE_API_BASE_URL}/playbook"
GENERATE_ENDPOINT = f"{ACE_API_BASE_URL}/generate"

# Dify机器人配置（示例）
DIFY_API_BASE_URL = "https://api.dify.ai/v1"
DIFY_API_KEY = "your_dify_api_key_here"

# 缓存配置
CACHE_SIZE = 10  # 缓存的对话数量
TRAINING_INTERVAL = 3600  # 训练间隔（秒），1小时

# ==================== 配置结束 ====================

class DifyACEIntegration:
    """Dify机器人与ACE训练API集成类"""
    
    def __init__(self, ace_api_url: str = ACE_API_BASE_URL):
        self.ace_api_url = ace_api_url
        self.dialogue_cache = []
        self.last_training_time = 0
        
    def add_dialogue(self, session_id: str, role: str, content: str):
        """添加对话到缓存"""
        timestamp = datetime.now().isoformat()
        dialogue = {
            "session_id": session_id,
            "role": role,  # "visitor" 或 "agent"
            "content": content,
            "timestamp": timestamp
        }
        
        self.dialogue_cache.append(dialogue)
        
        # 限制缓存大小
        if len(self.dialogue_cache) > CACHE_SIZE * 2:  # 每个对话包含visitor和agent两条消息
            self.dialogue_cache = self.dialogue_cache[-CACHE_SIZE * 2:]
        
        print(f"添加对话: {role} - {content[:50]}...")
    
    def should_train(self) -> bool:
        """判断是否应该触发训练"""
        current_time = time.time()
        time_since_last_training = current_time - self.last_training_time
        
        # 检查时间间隔和对话数量
        if time_since_last_training >= TRAINING_INTERVAL and len(self.dialogue_cache) >= CACHE_SIZE:
            return True
        
        return False
    
    def format_dialogues_for_training(self) -> List[Dict[str, Any]]:
        """格式化对话数据用于训练"""
        # 按会话ID分组
        dialogues_by_session = {}
        for dialogue in self.dialogue_cache:
            session_id = dialogue["session_id"]
            if session_id not in dialogues_by_session:
                dialogues_by_session[session_id] = []
            dialogues_by_session[session_id].append({
                "role": dialogue["role"],
                "content": dialogue["content"]
            })
        
        # 转换为训练请求格式
        formatted_dialogues = []
        for session_id, messages in dialogues_by_session.items():
            formatted_dialogues.append({
                "session_id": session_id,
                "messages": messages,
                "timestamp": datetime.now().isoformat()
            })
        
        return formatted_dialogues
    
    def trigger_training(self, task_type: str = "customer_service") -> Dict[str, Any]:
        """触发ACE模型训练"""
        try:
            # 格式化对话数据
            dialogues = self.format_dialogues_for_training()
            
            # 发送训练请求
            training_request = {
                "dialogues": dialogues,
                "task_type": task_type
            }
            
            print(f"发送训练请求，对话数量: {len(dialogues)}")
            response = requests.post(TRAIN_ENDPOINT, json=training_request)
            response.raise_for_status()
            
            result = response.json()
            self.last_training_time = time.time()
            
            # 清空缓存
            self.dialogue_cache = []
            
            print(f"训练请求已发送: {result['message']}")
            return result
            
        except Exception as e:
            print(f"触发训练失败: {str(e)}")
            return {"success": False, "message": f"触发训练失败: {str(e)}"}
    
    def get_updated_playbook(self, task_type: str = "customer_service") -> Optional[Dict[str, Any]]:
        """获取更新后的playbook"""
        try:
            response = requests.get(f"{PLAYBOOK_ENDPOINT}/{task_type}")
            response.raise_for_status()
            
            result = response.json()
            if result["success"]:
                return result["playbook"]
            else:
                print(f"获取playbook失败: {result['message']}")
                return None
                
        except Exception as e:
            print(f"获取playbook失败: {str(e)}")
            return None
    
    def generate_response_with_ace(self, question: str, task_type: str = "customer_service", context: str = "") -> Optional[str]:
        """使用ACE生成回复"""
        try:
            params = {
                "question": question,
                "context": context
            }
            
            response = requests.post(f"{GENERATE_ENDPOINT}/{task_type}", params=params)
            response.raise_for_status()
            
            result = response.json()
            if result["success"]:
                if task_type == "customer_service":
                    return result["response"]
                else:
                    return result["intent"]
            else:
                print(f"生成回复失败")
                return None
                
        except Exception as e:
            print(f"生成回复失败: {str(e)}")
            return None
    
    def process_dify_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理Dify webhook数据"""
        try:
            # 从webhook数据中提取对话信息
            # 注意：这里需要根据实际的Dify webhook格式进行调整
            session_id = webhook_data.get("conversation_id", "unknown")
            
            # 提取用户消息
            if "query" in webhook_data:
                self.add_dialogue(session_id, "visitor", webhook_data["query"])
            
            # 提取机器人回复
            if "answer" in webhook_data:
                self.add_dialogue(session_id, "agent", webhook_data["answer"])
            
            # 检查是否需要触发训练
            if self.should_train():
                training_result = self.trigger_training()
                return {
                    "status": "training_triggered",
                    "message": training_result["message"]
                }
            
            return {"status": "dialogue_cached", "cached_dialogues": len(self.dialogue_cache)}
            
        except Exception as e:
            print(f"处理webhook失败: {str(e)}")
            return {"status": "error", "message": str(e)}

# 示例使用
def example_usage():
    """示例使用DifyACEIntegration"""
    # 创建集成实例
    dify_ace = DifyACEIntegration()
    
    # 模拟添加对话
    dify_ace.add_dialogue("session_1", "visitor", "你们诊所有什么牙齿美白项目？")
    dify_ace.add_dialogue("session_1", "agent", "我们提供冷光美白和家庭美白两种方式。冷光美白效果明显，一次治疗可以提升5-8个色阶...")
    
    dify_ace.add_dialogue("session_2", "visitor", "牙齿矫正大概需要多长时间？")
    dify_ace.add_dialogue("session_2", "agent", "牙齿矫正的时间因人而异，通常需要1-3年。具体时间取决于您的牙齿情况和选择的矫正方式...")
    
    dify_ace.add_dialogue("session_3", "visitor", "拔牙后多久可以种植？")
    dify_ace.add_dialogue("session_3", "agent", "一般建议拔牙后3-6个月进行种植，等待牙槽骨愈合。但具体情况需要根据您的口腔状况来决定...")
    
    # 检查是否应该训练（在这个例子中，我们强制触发训练）
    if dify_ace.should_train() or True:  # 强制触发训练用于演示
        # 触发训练
        training_result = dify_ace.trigger_training()
        print(f"训练结果: {training_result}")
        
        # 等待一段时间让训练完成
        time.sleep(5)
        
        # 获取更新后的playbook
        updated_playbook = dify_ace.get_updated_playbook()
        if updated_playbook:
            print(f"获取到更新后的playbook，包含{len(updated_playbook.get('bullets', []))}条策略")
        
        # 使用ACE生成回复
        test_question = "你们诊所有什么牙齿美白项目？"
        ace_response = dify_ace.generate_response_with_ace(test_question)
        if ace_response:
            print(f"ACE生成的回复: {ace_response}")

# 创建Flask应用用于接收Dify webhook
from flask import Flask, request, jsonify

app = Flask(__name__)
dify_ace_integration = DifyACEIntegration()

@app.route('/webhook', methods=['POST'])
def dify_webhook():
    """接收Dify webhook的端点"""
    try:
        webhook_data = request.json
        result = dify_ace_integration.process_dify_webhook(webhook_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/manual_train', methods=['POST'])
def manual_train():
    """手动触发训练的端点"""
    try:
        task_type = request.json.get("task_type", "customer_service")
        result = dify_ace_integration.trigger_training(task_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_playbook/<task_type>', methods=['GET'])
def get_playbook(task_type):
    """获取playbook的端点"""
    try:
        playbook = dify_ace_integration.get_updated_playbook(task_type)
        if playbook:
            return jsonify({"success": True, "playbook": playbook})
        else:
            return jsonify({"success": False, "message": "获取playbook失败"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_response():
    """生成回复的端点"""
    try:
        data = request.json
        question = data.get("question")
        task_type = data.get("task_type", "customer_service")
        context = data.get("context", "")
        
        if not question:
            return jsonify({"success": False, "message": "缺少question参数"}), 400
        
        response = dify_ace_integration.generate_response_with_ace(question, task_type, context)
        if response:
            return jsonify({"success": True, "response": response})
        else:
            return jsonify({"success": False, "message": "生成回复失败"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dify与ACE集成服务")
    parser.add_argument("--mode", choices=["example", "server"], default="server", 
                       help="运行模式: example(运行示例) 或 server(启动Flask服务器)")
    args = parser.parse_args()
    
    if args.mode == "example":
        print("运行DifyACEIntegration示例...")
        example_usage()
    else:
        print("启动Dify与ACE集成Webhook服务器...")
        print("Webhook端点: http://localhost:5000/webhook")
        print("手动训练端点: http://localhost:5000/manual_train")
        print("获取playbook端点: http://localhost:5000/get_playbook/<task_type>")
        print("生成回复端点: http://localhost:5000/generate")
        app.run(host="0.0.0.0", port=5000, debug=True)