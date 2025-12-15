#!/usr/bin/env python3
"""
将对话流数据转换为ACE训练格式并更新playbook

从all_conversations.jsonl中读取对话数据，转换为ACE训练样本格式，
并使用这些样本训练更新kouqiang_customer_service_playbook.json
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# 添加项目路径
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from ace import Playbook, Sample, OfflineAdapter, Generator, Reflector, Curator
from ace.adaptation import TaskEnvironment, EnvironmentResult
from ace.llm_providers import LiteLLMClient

# ==================== 全局配置 ====================
# 测试模式配置 - 可以在这里修改样本量
TEST_MODE = False  # 是否为测试模式，设置为False切换到正式模式
MAX_TRAINING_SAMPLES = 10 if TEST_MODE else 50  # 测试模式使用10个样本，正式模式使用50个样本
TRAINING_EPOCHS = 1 if TEST_MODE else 1  # 测试模式使用1个epoch，正式模式使用3个epoch

# ==================== 全局配置结束 ====================


class ConversationEnvironment:
    """对话任务环境"""
    
    def __init__(self):
        pass
    
    def evaluate(self, sample: Sample, generator_output):
        """评估客服回复质量"""
        predicted_answer = generator_output.final_answer
        ground_truth = sample.ground_truth
        
        # 基础评估指标
        feedback = ""
        metrics = {}
        
        # 客服回复质量评估
        score = 0
        feedback_parts = []
        
        # 1. 相关性检查
        if any(keyword in predicted_answer for keyword in ["口腔", "牙齿", "服务", "预约"]):
            score += 0.3
            feedback_parts.append("回复内容相关性强")
        else:
            feedback_parts.append("回复内容相关性较弱")
        
        # 2. 完整性检查
        if len(predicted_answer) > 10:
            score += 0.2
            feedback_parts.append("回复内容完整")
        else:
            feedback_parts.append("回复内容过于简短")
        
        # 3. 专业性检查
        professional_keywords = ["检查", "方案", "治疗", "优惠", "预约"]
        if any(keyword in predicted_answer for keyword in professional_keywords):
            score += 0.3
            feedback_parts.append("回复具有专业性")
        
        # 4. 礼貌性检查
        polite_keywords = ["您好", "请问", "感谢", "稍后"]
        if any(keyword in predicted_answer for keyword in polite_keywords):
            score += 0.2
            feedback_parts.append("回复礼貌得体")
        
        feedback = f"客服回复评分: {score:.1f}/1.0. " + "; ".join(feedback_parts)
        
        # 计算布尔值用于metrics
        has_professional = any(keyword in predicted_answer for keyword in professional_keywords)
        is_polite = any(keyword in predicted_answer for keyword in polite_keywords)
        
        metrics = {
            "relevance_score": score,
            "length": len(predicted_answer),
            "has_professional_terms": 1.0 if has_professional else 0.0,
            "is_polite": 1.0 if is_polite else 0.0
        }
        
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics=metrics
        )


def load_conversations_from_jsonl(file_path: str) -> List[Dict]:
    """从JSONL文件加载对话数据"""
    conversations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            conversations.append(data)
    
    return conversations


def convert_to_samples(conversations: List[Dict]) -> List[Sample]:
    """将对话数据转换为ACE训练样本"""
    samples = []
    
    # 按conversation_id分组
    conversation_groups = {}
    for conv in conversations:
        conv_id = conv["conversation_id"]
        if conv_id not in conversation_groups:
            conversation_groups[conv_id] = []
        conversation_groups[conv_id].append(conv)
    
    # 为每个对话组创建样本
    for conv_id, messages in conversation_groups.items():
        # 按created_at排序
        messages.sort(key=lambda x: x.get("created_at", 0))
        
        # 提取对话历史
        dialogue_history = []
        for msg in messages:
            if msg["query"] and msg["answer"]:  # 只包含有问答的消息
                dialogue_history.append(f"用户: {msg['query']}")
                dialogue_history.append(f"客服: {msg['answer']}")
        
        # 为每个有回答的消息创建样本
        for i, msg in enumerate(messages):
            if msg["query"] and msg["answer"] and msg["status"] == "normal":  # 只处理正常状态的消息
                # 获取之前的对话历史作为上下文
                context = "\n".join(dialogue_history[:i*2]) if i > 0 else ""
                
                # 创建训练样本
                sample = Sample(
                    question=msg["query"],
                    context=context,
                    ground_truth=msg["answer"],
                    metadata={
                        "task_type": "customer_service_reply",
                        "conversation_id": conv_id,
                        "message_id": msg["id"],
                        "intent": classify_intent(msg["query"])
                    }
                )
                samples.append(sample)
    
    return samples


def classify_intent(user_message: str) -> str:
    """分类用户意图"""
    message = user_message.lower()
    
    if any(keyword in message for keyword in ["介绍", "服务", "项目"]):
        return "服务咨询"
    elif any(keyword in message for keyword in ["价格", "多少钱", "费用", "收费"]):
        return "价格咨询"
    elif any(keyword in message for keyword in ["预约", "到店", "联系"]):
        return "预约服务"
    elif any(keyword in message for keyword in ["优惠", "活动", "套餐"]):
        return "优惠咨询"
    elif any(keyword in message for keyword in ["地址", "位置", "在哪"]):
        return "地址咨询"
    else:
        return "一般咨询"


def save_samples_to_jsonl(samples: List[Sample], output_file: str):
    """保存训练样本为JSONL格式"""
    print(f"💾 保存训练样本到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            # 转换为字典格式
            sample_dict = {
                "question": sample.question,
                "context": sample.context,
                "ground_truth": sample.ground_truth,
                "metadata": sample.metadata
            }
            f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(samples)} 个训练样本")


def train_and_update_playbook(samples: List[Sample]):
    """训练模型并更新playbook"""
    print("\n🤖 训练口腔客服回复生成模型")
    print("=" * 40)
    
    mode_text = "测试模式" if TEST_MODE else "正式模式"
    print(f"📊 加载了 {len(samples)} 个客服回复训练样本 ({mode_text})")
    
    # 创建ACE组件
    llm = LiteLLMClient(
        model="openai/qwen-max-latest",
        api_key="sk-25587b057d5242428bb940d44035b5fd",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
        max_tokens=500,
        timeout=60
    )
    
    # 使用自定义提示模板，更适合中文LLM
    custom_generator_prompt = """你是一个专业的口腔客服助手，必须使用提供的策略手册来回答用户问题。
应用相关的策略，避免已知的错误，并展示逐步的推理过程。

策略手册：
{playbook}

最近的反思：
{reflection}

用户问题：
{question}

额外上下文：
{context}

请返回一个紧凑的JSON对象：
{{
  "reasoning": "<逐步的思考过程>",
  "bullet_ids": ["<策略ID1>", "<策略ID2>", "..."],
  "final_answer": "<简洁的最终回答>"
}}
"""

    # 使用中文重试提示，更适合中文LLM
    chinese_retry_prompt = "\n\n重要：请只返回一个有效的JSON对象。正确转义所有引号或使用单引号。不要在JSON之外包含任何其他文本。"

    generator = Generator(llm, prompt_template=custom_generator_prompt)
    reflector = Reflector(llm)
    curator = Curator(llm)
    
    # 创建环境和适配器
    environment = ConversationEnvironment()
    adapter = OfflineAdapter(
        generator=generator,
        reflector=reflector,
        curator=curator
    )
    
    # 加载现有的playbook
    playbook_path = "d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\kouqiang_playbook\\kouqiang_playbook.json"
    if os.path.exists(playbook_path):
        print(f"📖 加载现有playbook: {playbook_path}")
        adapter.playbook = Playbook.load_from_file(playbook_path)
    
    # 限制样本数量
    limited_samples = samples[:MAX_TRAINING_SAMPLES] if len(samples) > MAX_TRAINING_SAMPLES else samples
    
    # 训练模型 - 使用全局配置的epochs数
    print(f"\n🚀 开始训练 ({mode_text})...")
    adapter.run(limited_samples, environment, epochs=TRAINING_EPOCHS)  # 使用全局配置的epochs数
    
    # 保存playbook
    adapter.playbook.save_to_file(playbook_path)
    print(f"💾 训练完成，playbook已更新到: {playbook_path}")
    
    return adapter


def test_updated_playbook(adapter):
    """测试更新后的playbook"""
    print(f"\n🧪 测试更新后的playbook")
    print("=" * 30)
    
    test_cases = [
        "种牙多少钱",
        "全口种牙",
        "你好",
        "你们有什么服务？",
        "我想预约到店检查"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case}")
        
        # 生成回复
        sample = Sample(question=test_case, context="")
        result = adapter.generator.generate(
            question=sample.question,
            context=sample.context,
            playbook=adapter.playbook,
            reflection=""
        )
        
        print(f"回复: {result.final_answer}")
        print("-" * 50)


def main():
    """主函数"""
    mode_text = "测试模式" if TEST_MODE else "正式模式"
    print(f"🦷 对话流数据ACE训练系统 ({mode_text})")
    print("=" * 50)
    
    # 配置环境
    load_dotenv()
    
    # 创建输出目录
    output_dir = Path("d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\data_kouqiang_updata")
    output_dir.mkdir(exist_ok=True)
    
    # 步骤1: 加载对话数据
    print(f"\n📋 步骤1: 加载对话数据")
    print("=" * 30)
    
    data_file = "d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\conversations\\all_conversations.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    conversations = load_conversations_from_jsonl(data_file)
    print(f"📊 加载了 {len(conversations)} 条对话记录")
    
    # 步骤2: 转换为ACE训练样本
    print(f"\n🔄 步骤2: 转换为ACE训练样本")
    print("=" * 30)
    
    samples = convert_to_samples(conversations)
    print(f"📊 生成了 {len(samples)} 个训练样本")
    
    # 保存样本
    samples_file = str(output_dir / "conversation_samples.jsonl")
    save_samples_to_jsonl(samples, samples_file)
    
    # 步骤3: 训练模型并更新playbook
    print(f"\n🤖 步骤3: 训练模型并更新playbook")
    print("=" * 30)
    
    adapter = train_and_update_playbook(samples)
    
    # 步骤4: 测试更新后的playbook
    print(f"\n🧪 步骤4: 测试更新后的playbook")
    print("=" * 30)
    
    test_updated_playbook(adapter)
    
    print(f"\n🎉 {mode_text}训练完成！")
    print("📁 生成的文件:")
    print(f"  - {samples_file}")
    print("  - kouqiang_playbook.json (已更新)")


if __name__ == "__main__":
    main()