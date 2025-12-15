#!/usr/bin/env python3
"""
口腔客服对话数据转换为ACE训练样本

将dialogues_only_texts.txt中的对话转换为ACE Sample格式，
支持多种训练任务：客服回复生成、意图识别、对话管理等。
"""

import json
import re
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# 添加项目路径
# sys.path.append("d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main")

try:
    from ace import Sample
except ImportError:
    # 如果无法导入，使用简单的Sample类
    @dataclass
    class Sample:
        question: str
        context: str = ""
        ground_truth: Optional[str] = None
        metadata: Dict[str, object] = None


@dataclass
class DialogueSession:
    """对话会话"""
    session_id: int
    messages: List[Dict[str, str]]  # {"role": "visitor/robot", "content": "..."}
    metadata: Dict[str, object]


class KouQiangDataProcessor:
    """口腔对话数据处理器"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.sessions = []
        
    def load_and_parse(self) -> List[DialogueSession]:
        """加载并解析对话数据"""
        print("📖 加载口腔对话数据...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_session = []
        session_id = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 解析消息格式: "visitor: 内容" 或 "robot: 内容"
            if line.startswith('visitor:'):
                content = line[8:].strip()
                current_session.append({"role": "visitor", "content": content})
            elif line.startswith('robot:'):
                content = line[6:].strip()
                current_session.append({"role": "robot", "content": content})
            elif line.startswith('手机号：'):
                # 特殊处理手机号信息
                current_session.append({"role": "system", "content": line})
            else:
                # 其他格式，作为visitor消息处理
                current_session.append({"role": "visitor", "content": line})
        
        # 按对话轮次分组
        sessions = self._group_into_sessions(current_session)
        print(f"✅ 解析完成，共 {len(sessions)} 个对话会话")
        return sessions
    
    def _group_into_sessions(self, messages: List[Dict]) -> List[DialogueSession]:
        """将消息分组为对话会话"""
        sessions = []
        current_session = []
        session_id = 0
        
        # 简单的会话分割逻辑：以visitor开始新会话
        for msg in messages:
            if msg["role"] == "visitor" and current_session:
                # 结束当前会话，开始新会话
                if len(current_session) > 1:  # 至少包含一问一答
                    sessions.append(DialogueSession(
                        session_id=session_id,
                        messages=current_session.copy(),
                        metadata={"session_length": len(current_session)}
                    ))
                    session_id += 1
                current_session = [msg]
            else:
                current_session.append(msg)
        
        # 添加最后一个会话
        if len(current_session) > 1:
            sessions.append(DialogueSession(
                session_id=session_id,
                messages=current_session,
                metadata={"session_length": len(current_session)}
            ))
        
        return sessions
    
    def create_customer_service_samples(self, sessions: List[DialogueSession]) -> List[Sample]:
        """创建客服回复生成训练样本"""
        samples = []
        
        print("🤖 生成客服回复训练样本...")
        
        for session in sessions:
            # 提取对话历史
            dialogue_history = self._format_dialogue_history(session.messages)
            
            # 找到visitor的问题和robot的最佳回复
            for i, msg in enumerate(session.messages):
                if msg["role"] == "visitor" and i + 1 < len(session.messages):
                    next_msg = session.messages[i + 1]
                    if next_msg["role"] == "robot":
                        # 创建训练样本
                        sample = Sample(
                            question=msg["content"],
                            context=dialogue_history,
                            ground_truth=next_msg["content"],
                            metadata={
                                "task_type": "customer_service_reply",
                                "session_id": session.session_id,
                                "dialogue_turn": i,
                                "intent": self._classify_intent(msg["content"])
                            }
                        )
                        samples.append(sample)
        
        print(f"✅ 生成 {len(samples)} 个客服回复样本")
        return samples
    
    def create_intent_classification_samples(self, sessions: List[DialogueSession]) -> List[Sample]:
        """创建意图识别训练样本"""
        samples = []
        
        print("🎯 生成意图识别训练样本...")
        
        for session in sessions:
            for msg in session.messages:
                if msg["role"] == "visitor":
                    intent = self._classify_intent(msg["content"])
                    
                    sample = Sample(
                        question=msg["content"],
                        context="",
                        ground_truth=intent,
                        metadata={
                            "task_type": "intent_classification",
                            "session_id": session.session_id,
                            "intent_category": intent
                        }
                    )
                    samples.append(sample)
        
        print(f"✅ 生成 {len(samples)} 个意图识别样本")
        return samples
    
    def create_conversation_management_samples(self, sessions: List[DialogueSession]) -> List[Sample]:
        """创建对话管理训练样本"""
        samples = []
        
        print("🎭 生成对话管理训练样本...")
        
        for session in sessions:
            if len(session.messages) >= 4:  # 至少两轮对话
                # 分析对话状态和下一步动作
                dialogue_state = self._analyze_dialogue_state(session.messages)
                
                sample = Sample(
                    question=f"当前对话状态：{dialogue_state['description']}",
                    context=self._format_dialogue_history(session.messages),
                    ground_truth=dialogue_state["next_action"],
                    metadata={
                        "task_type": "conversation_management",
                        "session_id": session.session_id,
                        "dialogue_phase": dialogue_state["phase"],
                        "recommended_action": dialogue_state["next_action"]
                    }
                )
                samples.append(sample)
        
        print(f"✅ 生成 {len(samples)} 个对话管理样本")
        return samples
    
    def _format_dialogue_history(self, messages: List[Dict]) -> str:
        """格式化对话历史"""
        history = []
        for msg in messages[-6:]:  # 只保留最近6条消息作为上下文
            role = "用户" if msg["role"] == "visitor" else "客服"
            history.append(f"{role}: {msg['content']}")
        return "\n".join(history)
    
    def _classify_intent(self, user_message: str) -> str:
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
    
    def _analyze_dialogue_state(self, messages: List[Dict]) -> Dict:
        """分析对话状态"""
        # 简单的对话状态分析
        last_visitor_msg = None
        last_robot_msg = None
        
        for msg in reversed(messages):
            if msg["role"] == "visitor" and not last_visitor_msg:
                last_visitor_msg = msg["content"]
            elif msg["role"] == "robot" and not last_robot_msg:
                last_robot_msg = msg["content"]
        
        # 判断对话阶段
        if "联系方式" in last_robot_msg or "手机号" in last_robot_msg:
            phase = "信息收集"
            next_action = "等待用户提供联系方式"
        elif "预约" in last_visitor_msg or "到店" in last_visitor_msg:
            phase = "预约阶段"
            next_action = "安排预约时间和联系方式"
        elif "价格" in last_visitor_msg or "多少钱" in last_visitor_msg:
            phase = "价格咨询"
            next_action = "提供详细价格信息和优惠活动"
        else:
            phase = "初步咨询"
            next_action = "了解具体需求并提供相应服务介绍"
        
        return {
            "phase": phase,
            "description": f"用户最后询问：{last_visitor_msg[:50]}...",
            "next_action": next_action
        }
    
    def save_samples(self, samples: List[Sample], output_file: str):
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


def main():
    """主函数"""
    print("🦷 口腔客服对话数据ACE训练样本生成器")
    print("=" * 50)
    
    # 初始化处理器
    data_file = "d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\data_kouqiang\\dialogues_only_texts.txt"
    processor = KouQiangDataProcessor(data_file)
    
    # 加载和解析数据
    sessions = processor.load_and_parse()
    
    # 生成不同类型的训练样本
    all_samples = []
    
    # 1. 客服回复生成样本
    service_samples = processor.create_customer_service_samples(sessions)
    all_samples.extend(service_samples)
    
    # 2. 意图识别样本
    intent_samples = processor.create_intent_classification_samples(sessions)
    all_samples.extend(intent_samples)
    
    # 3. 对话管理样本
    management_samples = processor.create_conversation_management_samples(sessions)
    all_samples.extend(management_samples)
    
    # 保存训练样本
    output_dir = Path("d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\data_kouqiang")
    output_dir.mkdir(exist_ok=True)
    
    # 分别保存不同类型的样本
    processor.save_samples(service_samples, str(output_dir / "customer_service_samples.jsonl"))
    processor.save_samples(intent_samples, str(output_dir / "intent_classification_samples.jsonl"))
    processor.save_samples(management_samples, str(output_dir / "conversation_management_samples.jsonl"))
    
    # 保存合并的样本
    processor.save_samples(all_samples, str(output_dir / "kouqiang_all_samples.jsonl"))
    
    print(f"\n📊 样本统计:")
    print(f"  客服回复样本: {len(service_samples)}")
    print(f"  意图识别样本: {len(intent_samples)}")
    print(f"  对话管理样本: {len(management_samples)}")
    print(f"  总样本数: {len(all_samples)}")
    
    print(f"\n🎉 训练样本生成完成！")
    print(f"📁 输出目录: {output_dir}")


if __name__ == "__main__":
    main()