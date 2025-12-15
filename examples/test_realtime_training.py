#!/usr/bin/env python3
"""
测试ACE实时训练系统

验证ACE训练API和Dify-ACE集成服务是否正常工作
"""

import json
import time
import requests
from typing import Dict, Any

# 配置
ACE_API_BASE_URL = "http://localhost:8000"
DIFY_ACE_BASE_URL = "http://localhost:5000"

def test_ace_api():
    """测试ACE训练API"""
    print("=== 测试ACE训练API ===")
    
    try:
        # 1. 健康检查
        print("1. 健康检查...")
        response = requests.get(f"{ACE_API_BASE_URL}/health")
        response.raise_for_status()
        health_data = response.json()
        print(f"   状态: {health_data['status']}")
        print(f"   客服适配器: {health_data['customer_service_adapter']}")
        print(f"   意图识别适配器: {health_data['intent_adapter']}")
        
        # 2. 获取客服playbook
        print("\n2. 获取客服playbook...")
        response = requests.get(f"{ACE_API_BASE_URL}/playbook/customer_service")
        response.raise_for_status()
        playbook_data = response.json()
        print(f"   成功: {playbook_data['success']}")
        if playbook_data['success']:
            print(f"   策略数量: {len(playbook_data['playbook'].get('bullets', []))}")
        
        # 3. 生成回复
        print("\n3. 生成回复...")
        params = {
            "question": "你们诊所有什么牙齿美白项目？",
            "context": ""
        }
        response = requests.post(f"{ACE_API_BASE_URL}/generate/customer_service", params=params)
        response.raise_for_status()
        generate_data = response.json()
        print(f"   成功: {generate_data['success']}")
        if generate_data['success']:
            print(f"   回复: {generate_data['response'][:100]}...")
        
        # 4. 训练模型
        print("\n4. 训练模型...")
        training_request = {
            "dialogues": [
                {
                    "session_id": "test_session_1",
                    "messages": [
                        {"role": "visitor", "content": "你们诊所有什么牙齿美白项目？"},
                        {"role": "agent", "content": "我们提供冷光美白和家庭美白两种方式。冷光美白效果明显，一次治疗可以提升5-8个色阶。"}
                    ]
                },
                {
                    "session_id": "test_session_2",
                    "messages": [
                        {"role": "visitor", "content": "牙齿矫正大概需要多长时间？"},
                        {"role": "agent", "content": "牙齿矫正的时间因人而异，通常需要1-3年。具体时间取决于您的牙齿情况和选择的矫正方式。"}
                    ]
                }
            ],
            "task_type": "customer_service"
        }
        response = requests.post(f"{ACE_API_BASE_URL}/train", json=training_request)
        response.raise_for_status()
        training_data = response.json()
        print(f"   成功: {training_data['success']}")
        print(f"   消息: {training_data['message']}")
        
        print("\n✅ ACE训练API测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ ACE训练API测试失败: {str(e)}")
        return False

def test_dify_ace_integration():
    """测试Dify-ACE集成服务"""
    print("\n=== 测试Dify-ACE集成服务 ===")
    
    try:
        # 1. 手动触发训练
        print("1. 手动触发训练...")
        training_request = {"task_type": "customer_service"}
        response = requests.post(f"{DIFY_ACE_BASE_URL}/manual_train", json=training_request)
        response.raise_for_status()
        training_data = response.json()
        print(f"   成功: {training_data['success']}")
        print(f"   消息: {training_data['message']}")
        
        # 2. 获取playbook
        print("\n2. 获取playbook...")
        response = requests.get(f"{DIFY_ACE_BASE_URL}/get_playbook/customer_service")
        response.raise_for_status()
        playbook_data = response.json()
        print(f"   成功: {playbook_data['success']}")
        if playbook_data['success']:
            print(f"   策略数量: {len(playbook_data['playbook'].get('bullets', []))}")
        
        # 3. 生成回复
        print("\n3. 生成回复...")
        generate_request = {
            "question": "你们诊所有什么牙齿美白项目？",
            "task_type": "customer_service",
            "context": ""
        }
        response = requests.post(f"{DIFY_ACE_BASE_URL}/generate", json=generate_request)
        response.raise_for_status()
        generate_data = response.json()
        print(f"   成功: {generate_data['success']}")
        if generate_data['success']:
            print(f"   回复: {generate_data['response'][:100]}...")
        
        # 4. 模拟webhook
        print("\n4. 模拟webhook...")
        webhook_data = {
            "conversation_id": "test_webhook_session",
            "query": "拔牙后多久可以种植？",
            "answer": "一般建议拔牙后3-6个月进行种植，等待牙槽骨愈合。但具体情况需要根据您的口腔状况来决定。"
        }
        response = requests.post(f"{DIFY_ACE_BASE_URL}/webhook", json=webhook_data)
        response.raise_for_status()
        webhook_result = response.json()
        print(f"   状态: {webhook_result['status']}")
        if 'cached_dialogues' in webhook_result:
            print(f"   缓存的对话数量: {webhook_result['cached_dialogues']}")
        
        print("\n✅ Dify-ACE集成服务测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ Dify-ACE集成服务测试失败: {str(e)}")
        return False

def test_end_to_end():
    """端到端测试"""
    print("\n=== 端到端测试 ===")
    
    try:
        # 1. 添加多个对话到Dify-ACE集成服务
        print("1. 添加多个对话...")
        dialogues = [
            {
                "conversation_id": "e2e_session_1",
                "query": "你们诊所有什么牙齿美白项目？",
                "answer": "我们提供冷光美白和家庭美白两种方式。冷光美白效果明显，一次治疗可以提升5-8个色阶。"
            },
            {
                "conversation_id": "e2e_session_2",
                "query": "牙齿矫正大概需要多长时间？",
                "answer": "牙齿矫正的时间因人而异，通常需要1-3年。具体时间取决于您的牙齿情况和选择的矫正方式。"
            },
            {
                "conversation_id": "e2e_session_3",
                "query": "拔牙后多久可以种植？",
                "answer": "一般建议拔牙后3-6个月进行种植，等待牙槽骨愈合。但具体情况需要根据您的口腔状况来决定。"
            },
            {
                "conversation_id": "e2e_session_4",
                "query": "种植牙能用多久？",
                "answer": "种植牙的使用寿命因人而异，一般可以用10-20年甚至更长时间。良好的口腔卫生习惯和定期检查可以延长使用寿命。"
            },
            {
                "conversation_id": "e2e_session_5",
                "query": "儿童牙齿矫正最佳时期是什么时候？",
                "answer": "儿童牙齿矫正的最佳时期因情况而异。一般来说，功能性矫正可以在7-10岁进行，固定矫正通常在11-14岁，此时恒牙已基本萌出。"
            }
        ]
        
        for dialogue in dialogues:
            response = requests.post(f"{DIFY_ACE_BASE_URL}/webhook", json=dialogue)
            response.raise_for_status()
            result = response.json()
            print(f"   对话 {dialogue['conversation_id']}: {result['status']}")
        
        # 2. 手动触发训练
        print("\n2. 手动触发训练...")
        training_request = {"task_type": "customer_service"}
        response = requests.post(f"{DIFY_ACE_BASE_URL}/manual_train", json=training_request)
        response.raise_for_status()
        training_data = response.json()
        print(f"   训练请求: {training_data['success']}")
        
        # 3. 等待训练完成
        print("\n3. 等待训练完成...")
        time.sleep(10)  # 等待10秒让训练完成
        
        # 4. 获取更新后的playbook
        print("\n4. 获取更新后的playbook...")
        response = requests.get(f"{DIFY_ACE_BASE_URL}/get_playbook/customer_service")
        response.raise_for_status()
        playbook_data = response.json()
        print(f"   成功: {playbook_data['success']}")
        if playbook_data['success']:
            print(f"   策略数量: {len(playbook_data['playbook'].get('bullets', []))}")
        
        # 5. 使用更新后的模型生成回复
        print("\n5. 使用更新后的模型生成回复...")
        test_questions = [
            "你们诊所有什么牙齿美白项目？",
            "牙齿矫正大概需要多长时间？",
            "拔牙后多久可以种植？"
        ]
        
        for question in test_questions:
            generate_request = {
                "question": question,
                "task_type": "customer_service",
                "context": ""
            }
            response = requests.post(f"{DIFY_ACE_BASE_URL}/generate", json=generate_request)
            response.raise_for_status()
            generate_data = response.json()
            print(f"   问题: {question}")
            print(f"   回复: {generate_data['response'][:100]}...")
            print()
        
        print("✅ 端到端测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 端到端测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("开始测试ACE实时训练系统...")
    
    # 检查服务是否运行
    ace_api_running = False
    dify_ace_running = False
    
    try:
        response = requests.get(f"{ACE_API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            ace_api_running = True
            print("✅ ACE训练API正在运行")
    except:
        print("❌ ACE训练API未运行，请先启动: python examples/kouqiang_ace_training_api.py")
    
    try:
        response = requests.get(f"{DIFY_ACE_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            dify_ace_running = True
            print("✅ Dify-ACE集成服务正在运行")
    except:
        print("❌ Dify-ACE集成服务未运行，请先启动: python examples/dify_ace_integration.py")
    
    if not ace_api_running or not dify_ace_running:
        print("\n请先启动所有服务后再运行测试")
        return
    
    # 运行测试
    ace_api_test_passed = test_ace_api()
    dify_ace_test_passed = test_dify_ace_integration()
    
    if ace_api_test_passed and dify_ace_test_passed:
        print("\n=== 运行端到端测试 ===")
        test_end_to_end()
    
    print("\n测试完成")

if __name__ == "__main__":
    main()