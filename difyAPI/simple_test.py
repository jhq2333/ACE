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
        print(f"请求URL: {url}")
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text[:200]}...")
        print(f"获取对话列表失败: {e}")
        return None

def get_conversation_history(conversation_id: str, limit: int = 100) -> Optional[dict]:
    """获取指定对话ID的历史记录"""
    url = f"{DIFY_HOST}/v1/messages"
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
    
    # 创建合并的JSONL文件
    print(f"\n正在保存对话历史到 {output_dir}/all_conversations.jsonl...")
    with open(f"{output_dir}/all_conversations.jsonl", 'w', encoding='utf-8') as outfile:
        for i, conversation_id in enumerate(conversation_ids):
            print(f"[{i+1}/{len(conversation_ids)}] 正在获取对话 {conversation_id} 的历史记录...")
            history = get_conversation_history(conversation_id)
            
            if history and 'data' in history:
                for message in history['data']:
                    message['conversation_id'] = conversation_id
                    outfile.write(json.dumps(message, ensure_ascii=False) + '\n')
            else:
                print(f"跳过对话 {conversation_id}，无法获取历史记录")
    
    print(f"\n完成！所有对话历史已保存到 {output_dir}/all_conversations.jsonl")

if __name__ == "__main__":
    main()