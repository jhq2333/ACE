# main.py
import requests
from config_manager import config
import os
import json
import time
from typing import Dict, List, Optional


class DifyClient:
    def __init__(self):
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.conversation_history = []
    def call_workflow(self, inputs_data, query, user_id, response_mode, conversation_id=None, files=None, timeout=60):
        """Call a Dify workflow/chatflow.

        This method returns the requests.Response object on success. It raises
        a requests.RequestException on network errors so callers can handle it.
        """
        payload = {
            "inputs": inputs_data,
            "query": query,
            "response_mode": response_mode,
            "user": user_id,
            "conversation_id": conversation_id or ""
        }
        
        # 添加文件信息（如果有）
        if files:
            payload["files"] = files

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            # Only enable streaming at the requests level when response_mode == 'streaming'
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                stream=(response_mode == "streaming"),
                timeout=timeout,
            )
            # Raise for HTTP error statuses to make failures explicit
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            # Re-raise with a clearer message while preserving original exception
            raise RuntimeError(f"Failed to call Dify API: {exc}") from exc

    def call_chatflow(self, inputs_data, query, user_id, response_mode, conversation_id=None, files=None, timeout=60):
        """Compatibility alias used by example code: delegate to call_workflow."""
        return self.call_workflow(inputs_data, query, user_id, response_mode, conversation_id, files, timeout=timeout)
    
    def send_message(self, message: str, user_id: str, response_mode: str = "streaming", 
                    conversation_id: Optional[str] = None, files: Optional[List[Dict]] = None) -> Dict:
        """
        发送消息到 Dify 对话流并获取响应
        
        Args:
            message: 用户消息
            user_id: 用户ID
            response_mode: 响应模式 ("streaming" 或 "blocking")
            conversation_id: 对话ID，用于维持对话上下文
            files: 文件列表，格式为 [{"type": "image", "transfer_method": "remote_url", "url": "..."}]
            
        Returns:
            包含响应内容和元数据的字典
        """
        # 构建请求数据
        inputs = {}
            
        # 记录用户消息
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": time.time(),
            "files": files
        })
        
        try:
            # 调用 API
            response = self.call_chatflow(inputs, message, user_id, response_mode, conversation_id, files)
            
            # 处理响应
            result = {
                "success": True,
                "response_mode": response_mode,
                "status_code": response.status_code
            }
            
            if response_mode == "blocking":
                # 阻塞模式处理
                try:
                    response_data = response.json()
                    result["data"] = response_data
                    
                    # 提取并记录助手回复
                    if "answer" in response_data:
                        assistant_message = response_data["answer"]
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": assistant_message,
                            "timestamp": time.time()
                        })
                        result["message"] = assistant_message
                    
                    # 提取对话ID
                    if "conversation_id" in response_data:
                        result["conversation_id"] = response_data["conversation_id"]
                        
                except json.JSONDecodeError:
                    result["success"] = False
                    result["error"] = "Invalid JSON response"
                    result["raw_response"] = response.text
                    
            elif response_mode == "streaming":
                # 流式模式处理
                full_response = ""
                result["data"] = []
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        # 打印调试信息
                        print(f"\n[调试] 接收到的行: {decoded_line[:100]}...", flush=True)
                        
                        if decoded_line.startswith('data:'):
                            try:
                                # 去掉 "data:" 前缀并解析 JSON
                                data_str = decoded_line[len('data:'):].strip()
                                if data_str:
                                    data = json.loads(data_str)
                                    result["data"].append(data)
                                    
                                    # 打印事件类型
                                    if "event" in data:
                                        print(f"\n[调试] 事件类型: {data['event']}", flush=True)
                                    
                                    # 提取并累加 answer 字段
                                    if "answer" in data:
                                        answer_text = data["answer"]
                                        full_response += answer_text
                                        print(f"\n[调试] 收到回答片段: {answer_text[:50]}...", flush=True)
                                        
                                        # 提取对话ID
                                        if "conversation_id" in data and "conversation_id" not in result:
                                            result["conversation_id"] = data["conversation_id"]
                                            
                            except json.JSONDecodeError as e:
                                # 可能是流的结束标记或其他非 JSON 行
                                print(f"\n[调试] JSON解析错误: {e}", flush=True)
                                pass
                
                # 记录完整的助手回复
                if full_response:
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": time.time()
                    })
                    result["message"] = full_response
                    print(f"\n[调试] 完整回答长度: {len(full_response)}", flush=True)
                else:
                    print("\n[调试] 没有收到任何响应内容", flush=True)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_mode": response_mode
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史记录"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """清空对话历史记录"""
        self.conversation_history = []

def interactive_chat():
    """交互式对话功能"""
    client = DifyClient()
    user_id = "jhq"
    conversation_id = None
    
    print("=== Dify 对话流测试 ===")
    print("输入消息与 AI 对话，输入 'quit' 或 'exit' 退出，输入 'clear' 清空对话历史")
    print("输入 'history' 查看对话历史，输入 'mode' 切换响应模式")
    print("输入 'image' 添加图片文件，输入 'file' 添加其他文件")
    print("=" * 50)
    
    response_mode = "streaming"  # 默认使用流式模式
    files = []  # 当前会话的文件列表
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ").strip()
            
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            elif user_input.lower() == 'clear':
                client.clear_conversation_history()
                conversation_id = None
                files = []
                print("对话历史已清空")
                continue
            elif user_input.lower() == 'history':
                history = client.get_conversation_history()
                print("\n=== 对话历史 ===")
                for i, msg in enumerate(history):
                    role = "你" if msg["role"] == "user" else "AI"
                    print(f"{role}: {msg['content']}")
                    if msg["role"] == "user" and msg.get("files"):
                        print(f"  附件: {len(msg['files'])} 个文件")
                print("=" * 50)
                continue
            elif user_input.lower() == 'mode':
                response_mode = "blocking" if response_mode == "streaming" else "streaming"
                print(f"响应模式已切换为: {'阻塞模式' if response_mode == 'blocking' else '流式模式'}")
                continue
            elif user_input.lower() == 'image':
                url = input("请输入图片URL: ").strip()
                if url:
                    files.append({
                        "type": "image",
                        "transfer_method": "remote_url",
                        "url": url
                    })
                    print(f"已添加图片: {url}")
                continue
            elif user_input.lower() == 'file':
                url = input("请输入文件URL: ").strip()
                file_type = input("请输入文件类型 (image/file/等): ").strip() or "file"
                if url:
                    files.append({
                        "type": file_type,
                        "transfer_method": "remote_url",
                        "url": url
                    })
                    print(f"已添加文件: {url}")
                continue
            elif not user_input:
                print("请输入有效消息")
                continue
            
            # 发送消息并获取响应
            print(f"\nAI ({'流式' if response_mode == 'streaming' else '阻塞'}模式): ", end="", flush=True)
            
            # 如果有文件，显示文件信息
            if files:
                print(f"\n[已附加 {len(files)} 个文件]")
            
            result = client.send_message(user_input, user_id, response_mode, conversation_id, files)
            
            if result["success"]:
                # 更新对话ID
                if "conversation_id" in result:
                    conversation_id = result["conversation_id"]
                
                # 打印AI回复
                if "message" in result and result["message"]:
                    print(result["message"])
                else:
                    print("(AI没有回复)")
            else:
                print(f"\n错误: {result.get('error', '未知错误')}")
            
            # 清空当前文件列表（每个消息只使用一次）
            files = []
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")


def curl_example_test():
    """演示与 curl 命令等效的功能"""
    client = DifyClient()
    
    print("=== curl 命令等效测试 ===")
    print("这个测试模拟了你提供的 curl 命令的功能")
    print("=" * 50)
    
    # 示例1: 简单文本查询
    print("\n1. 简单文本查询:")
    query = "What are the specs of the iPhone 13 Pro Max?"
    result = client.send_message(
        message=query,
        user_id="abc-123",
        response_mode="streaming"
    )
    
    if result["success"]:
        print(f"问题: {query}")
        print(f"回答: {result.get('message', '无回复')}")
        if "conversation_id" in result:
            print(f"对话ID: {result['conversation_id']}")
    else:
        print(f"请求失败: {result.get('error', '未知错误')}")
    
    # 示例2: 带图片的查询
    print("\n2. 带图片的查询:")
    query = "请描述这张图片"
    image_url = "https://cloud.dify.ai/logo/logo-site.png"
    files = [
        {
            "type": "image",
            "transfer_method": "remote_url",
            "url": image_url
        }
    ]
    
    result = client.send_message(
        message=query,
        user_id="abc-123",
        response_mode="streaming",
        files=files
    )
    
    if result["success"]:
        print(f"问题: {query}")
        print(f"图片: {image_url}")
        print(f"回答: {result.get('message', '无回复')}")
        if "conversation_id" in result:
            print(f"对话ID: {result['conversation_id']}")
    else:
        print(f"请求失败: {result.get('error', '未知错误')}")
    
    # 示例3: 阻塞模式查询
    print("\n3. 阻塞模式查询:")
    query = "请用一句话总结你自己"
    result = client.send_message(
        message=query,
        user_id="abc-123",
        response_mode="blocking"
    )
    
    if result["success"]:
        print(f"问题: {query}")
        print(f"回答: {result.get('message', '无回复')}")
        if "conversation_id" in result:
            print(f"对话ID: {result['conversation_id']}")
    else:
        print(f"请求失败: {result.get('error', '未知错误')}")


def simple_test():
    """简单的测试函数，展示基本用法"""
    client = DifyClient()
    
    # 发送单条消息
    print("=== 简单测试 ===")
    result = client.send_message("你好，请介绍一下自己", "test_user", "streaming")
    
    if result["success"]:
        print(f"AI 回复: {result.get('message', '无回复')}")
        if "conversation_id" in result:
            print(f"对话ID: {result['conversation_id']}")
    else:
        print(f"请求失败: {result.get('error', '未知错误')}")
    
    # 查看对话历史
    history = client.get_conversation_history()
    print(f"\n对话历史记录数: {len(history)}")


# 使用示例
if __name__ == "__main__":
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 交互式对话 (推荐)")
    print("2. 简单测试")
    print("3. curl 命令等效测试")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    if choice == "1":
        interactive_chat()
    elif choice == "2":
        simple_test()
    elif choice == "3":
        curl_example_test()
    else:
        print("无效选择，运行简单测试")
        simple_test()