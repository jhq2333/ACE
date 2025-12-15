import os
import json
import requests
from typing import Optional, List

# 配置信息
DIFY_API_KEY = os.environ.get("DIFY_API_KEY", "app-qtsvkyrD6RAUW5LAHALVXUX0")
DIFY_HOST = os.environ.get("DIFY_HOST", "http://36.248.221.38")

def get_conversations_list(limit: int = 20) -> Optional[List[str]]:
    """获取对话ID列表"""
    url = f"{DIFY_HOST}/v1/conversations?user=jhq"
    headers = {"Authorization": f"Bearer {DIFY_API_KEY}"}
    params = {"limit": limit}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        print(f"请求URL: {url}")
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text[:200]}...")
        response.raise_for_status()
        data = response.json()
        
        # 提取对话ID列表
        conversation_ids = []
        if 'data' in data:
            for conv in data['data']:
                if 'id' in conv:
                    conversation_ids.append(conv['id'])
        
        return conversation_ids
    except requests.RequestException as e:
        print(f"获取对话列表失败: {e}")
        return None

def get_conversation_history(conversation_id: str, limit: int = 100) -> Optional[dict]:
    """获取指定对话ID的历史记录"""
    url = f"{DIFY_HOST}/v1/messages?user=jhq"
    headers = {"Authorization": f"Bearer {DIFY_API_KEY}"}
    params = {"conversation_id": conversation_id, "limit": limit}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        print(f"请求URL: {url}")
        print(f"状态码: {response.status_code}")
        if response.status_code != 200:
            print(f"响应内容: {response.text[:200]}...")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"获取对话 {conversation_id} 历史失败: {e}")
        return None

def save_to_jsonl(data: dict, conversation_id: str, filename: str):
    """将对话数据保存到JSONL文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        # 添加对话ID到每条记录
        if 'data' in data:
            for message in data['data']:
                message['conversation_id'] = conversation_id
                f.write(json.dumps(message, ensure_ascii=False) + '\n')
        print(f"对话 {conversation_id} 历史已保存到 {filename}")

def main():
    print("正在获取对话列表...")
    conversation_ids = get_conversations_list(20)
    
    if not conversation_ids:
        print("未能获取到对话列表")
        return
    
    print(f"找到 {len(conversation_ids)} 个对话")
    
    # 创建输出目录
    output_dir = "conversations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存所有对话ID到一个文件
    with open(f"{output_dir}/conversation_ids.txt", 'w', encoding='utf-8') as f:
        for conv_id in conversation_ids:
            f.write(f"{conv_id}\n")
    
    # 获取每个对话的历史记录
    for i, conversation_id in enumerate(conversation_ids):
        print(f"\n[{i+1}/{len(conversation_ids)}] 正在获取对话 {conversation_id} 的历史记录...")
        history = get_conversation_history(conversation_id)
        
        if history:
            filename = f"{output_dir}/conversation_{conversation_id}.jsonl"
            save_to_jsonl(history, conversation_id, filename)
        else:
            print(f"跳过对话 {conversation_id}，无法获取历史记录")
    
    # 创建合并的JSONL文件
    print("\n正在创建合并的JSONL文件...")
    with open(f"{output_dir}/all_conversations.jsonl", 'w', encoding='utf-8') as outfile:
        for conversation_id in conversation_ids:
            filename = f"{output_dir}/conversation_{conversation_id}.jsonl"
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
    
    print(f"\n完成！所有对话历史已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()