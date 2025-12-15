#!/usr/bin/env python3
"""
测试webhook服务器并创建dify_incoming.jsonl文件
"""
import requests
import json
import time

# Webhook服务器配置
WEBHOOK_URL = "http://localhost:8000/webhook"
# 尝试几个可能的API密钥
POSSIBLE_API_KEYS = ["app-qtsvkyrD6RAUW5LAHALVXUX0", "changeme", "", "test", "dify"]

# 测试数据
test_data = {
    "conversation_id": "test_conversation_001",
    "user": "test_user",
    "messages": [
        {"role": "user", "text": "你好，我想咨询一下牙齿矫正的问题"},
        {"role": "assistant", "text": "您好！关于牙齿矫正，我可以为您提供一些基本信息。请问您具体想了解哪方面的内容呢？"},
        {"role": "user", "text": "我想知道隐形矫正和传统钢牙套的区别"},
        {"role": "assistant", "text": "隐形矫正和传统钢牙套各有优势。隐形矫正美观度高，可以自行摘戴；传统钢牙矫正在某些复杂病例中效果更稳定。"}
    ]
}

def check_server_health():
    """检查webhook服务器健康状态"""
    try:
        # 尝试访问根路径
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✓ Webhook服务器正在运行")
            # 尝试访问文档页面
            docs_response = requests.get("http://localhost:8000/docs", timeout=5)
            if docs_response.status_code == 200:
                print("✓ API文档可访问")
            return True
        else:
            print(f"✗ Webhook服务器响应异常，状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到webhook服务器，请确保服务器已启动")
        print("运行命令: uvicorn difyAPI.webhook_server:app --reload --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"✗ 检查服务器健康状态时出错: {str(e)}")
        return False

def test_webhook():
    """测试webhook服务器"""
    # 首先检查服务器是否运行
    if not check_server_health():
        return
    
    # 尝试不同的API密钥
    for api_key in POSSIBLE_API_KEYS:
        headers = {
            "Content-Type": "application/json",
            "x-dify-signature": api_key
        }
        
        print(f"\n尝试使用API密钥: '{api_key}'")
        print(f"请求URL: {WEBHOOK_URL}")
        
        try:
            print("发送测试数据到webhook服务器...")
            response = requests.post(WEBHOOK_URL, json=test_data, headers=headers)
            
            if response.status_code == 200:
                print(f"✓ 使用API密钥 '{api_key}' 成功！")
                print(f"服务器响应: {response.json()}")
                print("\n等待几秒让后台工作进程处理数据...")
                time.sleep(3)
                
                # 检查文件是否创建
                import os
                file_path = os.path.join(os.path.dirname(__file__), "..", "dify_incoming.jsonl")
                file_path = os.path.abspath(file_path)
                
                if os.path.exists(file_path):
                    print(f"✓ dify_incoming.jsonl文件已创建: {file_path}")
                    print("\n文件内容预览:")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines[:3], 1):  # 只显示前3行
                            print(f"行 {i}: {line.strip()}")
                        if len(lines) > 3:
                            print(f"... (共 {len(lines)} 行)")
                    return  # 成功，退出函数
                else:
                    print("✗ dify_incoming.jsonl文件尚未创建，请检查webhook服务器是否正常运行")
            elif response.status_code == 404:
                print(f"✗ 端点不存在 (404)，请检查webhook服务器是否正确配置")
                print("可能的原因:")
                print("1. 服务器没有正确加载webhook端点")
                print("2. 端点路径不正确")
                print("3. 服务器没有正确启动")
                return  # 404错误不需要尝试其他API密钥
            else:
                print(f"✗ 请求失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("✗ 无法连接到webhook服务器，请确保服务器已启动")
            print("运行命令: uvicorn difyAPI.webhook_server:app --reload --host 0.0.0.0 --port 8000")
            break  # 连接错误，不需要尝试其他API密钥
        except Exception as e:
            print(f"✗ 发生错误: {str(e)}")
    
    print("\n所有API密钥都尝试失败。请检查webhook服务器的API密钥配置。")
    print("你可以通过设置环境变量来指定API密钥:")
    print("set DIFY_WEBHOOK_API_KEY=your_api_key")
    print("然后重启webhook服务器。")

if __name__ == "__main__":
    test_webhook()