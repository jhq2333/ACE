#!/usr/bin/env python3
"""
口腔客服ACE训练脚本

使用转换后的口腔对话数据训练ACE模型，
支持多种任务：客服回复生成、意图识别、对话管理。
"""

import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from ace import Playbook, Sample, OfflineAdapter, Generator, Reflector, Curator
from ace.adaptation import TaskEnvironment, EnvironmentResult
from ace.llm_providers import LiteLLMClient
import os
# 添加项目路径到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from examples.chatflow_kouqiang.kouqiang_data_converter import KouQiangDataProcessor

# ==================== 全局配置 ====================
# 测试模式配置 - 可以在这里修改样本量
TEST_MODE = False  # 是否为测试模式，设置为False切换到正式模式
MAX_SESSIONS = 20 if TEST_MODE else 100  # 测试模式使用20个会话，正式模式使用100个会话
MAX_TRAINING_SAMPLES = 10 if TEST_MODE else 50  # 测试模式使用10个样本，正式模式使用50个样本
TRAINING_EPOCHS = 1 if TEST_MODE else 3  # 测试模式使用1个epoch，正式模式使用3个epoch

# ==================== 全局配置结束 ====================


class KouQiangEnvironment:
    """口腔客服任务环境"""
    
    def __init__(self, task_type: str = "customer_service"):
        self.task_type = task_type
    
    def evaluate(self, sample: Sample, generator_output):
        """评估客服回复质量"""
        predicted_answer = generator_output.final_answer
        ground_truth = sample.ground_truth
        
        # 基础评估指标
        feedback = ""
        metrics = {}
        
        if self.task_type == "customer_service":
            # 客服回复质量评估
            feedback, metrics = self._evaluate_customer_service(
                predicted_answer, ground_truth, sample
            )
        elif self.task_type == "intent_classification":
            # 意图识别准确率评估
            feedback, metrics = self._evaluate_intent_classification(
                predicted_answer, ground_truth
            )
        elif self.task_type == "conversation_management":
            # 对话管理策略评估
            feedback, metrics = self._evaluate_conversation_management(
                predicted_answer, ground_truth, sample
            )
        
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics=metrics
        )
    
    def _evaluate_customer_service(self, predicted: str, ground_truth: str, sample: Sample):
        """评估客服回复质量"""
        # 简单的评估逻辑
        score = 0
        feedback_parts = []
        
        # 1. 相关性检查
        if any(keyword in predicted for keyword in ["口腔", "牙齿", "服务", "预约"]):
            score += 0.3
            feedback_parts.append("回复内容相关性强")
        else:
            feedback_parts.append("回复内容相关性较弱")
        
        # 2. 完整性检查
        if len(predicted) > 10:
            score += 0.2
            feedback_parts.append("回复内容完整")
        else:
            feedback_parts.append("回复内容过于简短")
        
        # 3. 专业性检查
        professional_keywords = ["检查", "方案", "治疗", "优惠", "预约"]
        if any(keyword in predicted for keyword in professional_keywords):
            score += 0.3
            feedback_parts.append("回复具有专业性")
        
        # 4. 礼貌性检查
        polite_keywords = ["您好", "请问", "感谢", "稍后"]
        if any(keyword in predicted for keyword in polite_keywords):
            score += 0.2
            feedback_parts.append("回复礼貌得体")
        
        feedback = f"客服回复评分: {score:.1f}/1.0. " + "; ".join(feedback_parts)
        
        # 计算布尔值用于metrics
        has_professional = any(keyword in predicted for keyword in professional_keywords)
        is_polite = any(keyword in predicted for keyword in polite_keywords)
        
        return feedback, {
            "relevance_score": score,
            "length": len(predicted),
            "has_professional_terms": 1.0 if has_professional else 0.0,
            "is_polite": 1.0 if is_polite else 0.0
        }
    
    def _evaluate_intent_classification(self, predicted: str, ground_truth: str):
        """评估意图识别准确率"""
        is_correct = predicted.strip() == ground_truth.strip()
        
        feedback = f"意图识别{'正确' if is_correct else '错误'}。"
        if not is_correct:
            feedback += f" 预测: {predicted}, 实际: {ground_truth}"
        
        return feedback, {
            "accuracy": 1.0 if is_correct else 0.0,
            "predicted_intent": predicted,
            "ground_truth_intent": ground_truth
        }
    
    def _evaluate_conversation_management(self, predicted: str, ground_truth: str, sample: Sample):
        """评估对话管理策略"""
        # 简单的策略匹配
        action_keywords = {
            "收集信息": ["联系方式", "手机号", "姓名"],
            "提供服务": ["介绍", "项目", "服务"],
            "安排预约": ["预约", "时间", "到店"],
            "提供优惠": ["优惠", "活动", "价格"]
        }
        
        predicted_action = "未知"
        for action, keywords in action_keywords.items():
            if any(keyword in predicted for keyword in keywords):
                predicted_action = action
                break
        
        is_relevant = predicted_action != "未知"
        
        feedback = f"对话策略{'合理' if is_relevant else '不合理'}。"
        if is_relevant:
            feedback += f" 识别动作: {predicted_action}"
        
        return feedback, {
            "action_relevance": 1.0 if is_relevant else 0.0,
            "predicted_action": predicted_action,
            "response_length": len(predicted)
        }


def load_samples_from_jsonl(file_path: str, max_samples: int = None) -> list:
    """从JSONL文件加载训练样本"""
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if max_samples and line_num >= max_samples:
                break
                
            data = json.loads(line.strip())
            sample = Sample(
                question=data["question"],
                context=data.get("context", ""),
                ground_truth=data.get("ground_truth"),
                metadata=data.get("metadata", {})
            )
            samples.append(sample)
    
    return samples


def train_customer_service_model():
    """训练客服回复生成模型"""
    print("\n🤖 训练口腔客服回复生成模型")
    print("=" * 40)
    
    # 加载训练数据 - 使用全局配置的样本量
    data_file = "d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\data_kouqiang\\customer_service_samples.jsonl"
    samples = load_samples_from_jsonl(data_file, max_samples=MAX_TRAINING_SAMPLES)
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

    generator = Generator(llm, prompt_template=custom_generator_prompt, retry_prompt=chinese_retry_prompt)
    reflector = Reflector(llm)
    curator = Curator(llm)
    
    # 创建环境和适配器
    environment = KouQiangEnvironment("customer_service")
    adapter = OfflineAdapter(
        generator=generator,
        reflector=reflector,
        curator=curator
    )
    
    # 训练模型 - 使用全局配置的epochs数
    mode_text = "测试模式" if TEST_MODE else "正式模式"
    print(f"\n🚀 开始训练 ({mode_text})...")
    adapter.run(samples, environment, epochs=TRAINING_EPOCHS)  # 使用全局配置的epochs数
    
    # 保存playbook
    playbook_path = "d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\kouqiang_customer_service_playbook.json"
    adapter.playbook.save_to_file(playbook_path)
    print(f"💾 训练完成，playbook已保存到: {playbook_path}")
    
    return adapter


def train_intent_classification_model():
    """训练意图识别模型"""
    print("\n🎯 训练口腔客服意图识别模型")
    print("=" * 40)
    
    # 加载训练数据 - 使用全局配置的样本量
    data_file = "d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\data_kouqiang\\intent_classification_samples.jsonl"
    samples = load_samples_from_jsonl(data_file, max_samples=MAX_TRAINING_SAMPLES)
    mode_text = "测试模式" if TEST_MODE else "正式模式"
    print(f"📊 加载了 {len(samples)} 个意图识别训练样本 ({mode_text})")
    
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
    custom_generator_prompt = """你是一个专业的口腔客服意图分类助手，必须使用提供的策略手册来识别用户意图。
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

    generator = Generator(llm, prompt_template=custom_generator_prompt, retry_prompt=chinese_retry_prompt)
    reflector = Reflector(llm)
    curator = Curator(llm)
    
    # 创建环境和适配器
    environment = KouQiangEnvironment("intent_classification")
    adapter = OfflineAdapter(
        generator=generator,
        reflector=reflector,
        curator=curator
    )
    
    # 训练模型 - 使用全局配置的epochs数
    mode_text = "测试模式" if TEST_MODE else "正式模式"
    print(f"\n🚀 开始训练 ({mode_text})...")
    adapter.run(samples, environment, epochs=TRAINING_EPOCHS)  # 使用全局配置的epochs数
    
    # 保存playbook
    playbook_path = "d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\kouqiang_intent_playbook.json"
    adapter.playbook.save_to_file(playbook_path)
    print(f"💾 训练完成，playbook已保存到: {playbook_path}")
    
    return adapter


def test_trained_model(adapter, task_name: str):
    """测试训练好的模型"""
    print(f"\n🧪 测试{task_name}模型")
    print("=" * 30)
    
    test_cases = [
        "你们有什么服务？",
        "种一颗牙多少钱？",
        "我想预约到店检查",
        "最近有什么优惠活动？",
        "你们的地址在哪里？"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case}")
        
        # 生成回复 - 使用adapter的generator，它已经配置了自定义提示模板
        sample = Sample(question=test_case, context="")
        result = adapter.generator.generate(
            question=sample.question,
            context=sample.context,
            playbook=adapter.playbook,
            reflection="",
            sample=sample
        )
        
        print(f"回复: {result.final_answer}")
        print("-" * 50)


def main():
    """主函数"""
    mode_text = "测试模式" if TEST_MODE else "正式模式"
    print(f"🦷 口腔客服ACE训练系统 ({mode_text})")
    print("=" * 50)
    
    # 配置环境
    load_dotenv()
    
    # # 配置Opik（如果可用）
    # try:
    #     configure_opik(project_name="kouqiang-ace-training")
    #     print("📊 Opik observability enabled")
    # except:
    #     print("📊 Opik not available, continuing without observability")
    
    # 创建数据目录
    output_dir = Path("d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\data_kouqiang")
    output_dir.mkdir(exist_ok=True)
    
    # 步骤1: 数据预处理 - 使用全局配置的会话数量
    mode_text = "测试模式" if TEST_MODE else "正式模式"
    print(f"\n📋 步骤1: 数据预处理 ({mode_text})")
    print("=" * 30)
    
    data_file = "d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\data_kouqiang\\dialogues_only_texts.txt"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    processor = KouQiangDataProcessor(data_file)
    sessions = processor.load_and_parse()
    
    # 生成训练样本 - 使用全局配置的会话数量
    limited_sessions = sessions[:MAX_SESSIONS] if len(sessions) > MAX_SESSIONS else sessions
    service_samples = processor.create_customer_service_samples(limited_sessions)
    intent_samples = processor.create_intent_classification_samples(limited_sessions)
    
    # 保存样本
    processor.save_samples(service_samples, str(output_dir / "customer_service_samples.jsonl"))
    processor.save_samples(intent_samples, str(output_dir / "intent_classification_samples.jsonl"))
    
    print(f"📊 使用了 {len(limited_sessions)} 个会话生成训练样本 ({mode_text})")
    
    # 步骤2: 模型训练
    print(f"\n🤖 步骤2: 模型训练 ({mode_text})")
    print("=" * 30)
    
    # 训练客服回复生成模型
    service_adapter = train_customer_service_model()
    test_trained_model(service_adapter, "客服回复生成")
    
    # 训练意图识别模型
    intent_adapter = train_intent_classification_model()
    test_trained_model(intent_adapter, "意图识别")
    
    mode_text = "测试" if TEST_MODE else "正式"
    print(f"\n🎉 {mode_text}训练完成！")
    print("📁 生成的文件:")
    print("  - kouqiang_customer_service_playbook.json")
    print("  - kouqiang_intent_playbook.json")
    print("  - customer_service_samples.jsonl")
    print("  - intent_classification_samples.jsonl")


if __name__ == "__main__":
    main()