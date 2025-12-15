#!/usr/bin/env python3
"""
口腔客服ACE训练HTTP API服务

提供RESTful API接口，用于：
1. 接收来自Dify机器人的对话数据
2. 实时训练ACE模型并更新playbook
3. 将更新后的playbook返回给Dify机器人
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 添加项目路径到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ace import Playbook, Sample, OfflineAdapter, Generator, Reflector, Curator
from ace.adaptation import TaskEnvironment, EnvironmentResult
from ace.llm_providers import LiteLLMClient
from examples.chatflow_kouqiang.kouqiang_data_converter import KouQiangDataProcessor
from examples.chatflow_kouqiang.kouqiang_ace_training import KouQiangEnvironment

# ==================== 全局配置 ====================
# API配置
API_HOST = "0.0.0.0"
API_PORT = 8000

# ACE训练配置
LLM_MODEL = "openai/qwen-max-latest"
API_KEY = "sk-25587b057d5242428bb940d44035b5fd"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEMPERATURE = 0.7
MAX_TOKENS = 500
TIMEOUT = 60

# 训练配置
BATCH_SIZE = 5  # 每批处理的对话数量
MIN_TRAINING_SAMPLES = 3  # 最小训练样本数

# 文件路径
DATA_DIR = Path("d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main\\data_kouqiang")
PLAYBOOK_DIR = Path("d:\\jhq\\agentic-context-engine-main\\agentic-context-engine-main")
CUSTOMER_SERVICE_PLAYBOOK = PLAYBOOK_DIR / "kouqiang_customer_service_playbook.json"
INTENT_PLAYBOOK = PLAYBOOK_DIR / "kouqiang_intent_playbook.json"

# ==================== 全局配置结束 ====================

# 创建FastAPI应用
app = FastAPI(
    title="口腔客服ACE训练API",
    description="用于实时训练ACE模型并更新playbook的API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储适配器
customer_service_adapter: Optional[OfflineAdapter] = None
intent_adapter: Optional[OfflineAdapter] = None

# 定义数据模型
class DialogueMessage(BaseModel):
    role: str  # "visitor" 或 "agent"
    content: str

class DialogueData(BaseModel):
    session_id: str
    messages: List[DialogueMessage]
    timestamp: Optional[str] = None

class TrainingRequest(BaseModel):
    dialogues: List[DialogueData]
    task_type: str = "customer_service"  # "customer_service" 或 "intent_classification"

class TrainingResponse(BaseModel):
    success: bool
    message: str
    playbook_updated: bool = False
    playbook_path: Optional[str] = None

class PlaybookResponse(BaseModel):
    success: bool
    playbook: Optional[Dict[str, Any]] = None
    message: str

# 初始化函数
def initialize_adapters():
    """初始化ACE适配器"""
    global customer_service_adapter, intent_adapter
    
    print("初始化ACE适配器...")
    
    # 创建LLM客户端
    llm = LiteLLMClient(
        model=LLM_MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT
    )
    
    # 自定义提示模板
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

    chinese_retry_prompt = "\n\n重要：请只返回一个有效的JSON对象。正确转义所有引号或使用单引号。不要在JSON之外包含任何其他文本。"
    
    # 创建ACE组件
    generator = Generator(llm, prompt_template=custom_generator_prompt, retry_prompt=chinese_retry_prompt)
    reflector = Reflector(llm)
    curator = Curator(llm)
    
    # 初始化客服适配器
    if CUSTOMER_SERVICE_PLAYBOOK.exists():
        print(f"加载现有客服playbook: {CUSTOMER_SERVICE_PLAYBOOK}")
        customer_service_playbook = Playbook.load_from_file(str(CUSTOMER_SERVICE_PLAYBOOK))
    else:
        print("创建新的客服playbook")
        customer_service_playbook = Playbook()
    
    customer_service_environment = KouQiangEnvironment("customer_service")
    customer_service_adapter = OfflineAdapter(
        generator=generator,
        reflector=reflector,
        curator=curator,
        playbook=customer_service_playbook
    )
    
    # 初始化意图识别适配器
    if INTENT_PLAYBOOK.exists():
        print(f"加载现有意图识别playbook: {INTENT_PLAYBOOK}")
        intent_playbook = Playbook.load_from_file(str(INTENT_PLAYBOOK))
    else:
        print("创建新的意图识别playbook")
        intent_playbook = Playbook()
    
    intent_environment = KouQiangEnvironment("intent_classification")
    intent_adapter = OfflineAdapter(
        generator=generator,
        reflector=reflector,
        curator=curator,
        playbook=intent_playbook
    )
    
    print("ACE适配器初始化完成")

def process_dialogues_to_samples(dialogues: List[DialogueData], task_type: str) -> List[Sample]:
    """将对话数据转换为训练样本"""
    samples = []
    
    for dialogue in dialogues:
        # 提取对话对
        messages = dialogue.messages
        for i in range(len(messages)):
            if messages[i].role == "visitor" and i+1 < len(messages) and messages[i+1].role == "agent":
                # 创建客服回复样本
                if task_type == "customer_service":
                    sample = Sample(
                        question=messages[i].content,
                        context="",
                        ground_truth=messages[i+1].content,
                        metadata={"session_id": dialogue.session_id}
                    )
                    samples.append(sample)
                
                # 创建意图识别样本
                elif task_type == "intent_classification":
                    # 简单的意图分类逻辑
                    intent = classify_intent(messages[i].content)
                    sample = Sample(
                        question=messages[i].content,
                        context="",
                        ground_truth=intent,
                        metadata={"session_id": dialogue.session_id}
                    )
                    samples.append(sample)
    
    return samples

def classify_intent(question: str) -> str:
    """简单的意图分类函数"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["服务", "介绍", "项目"]):
        return "服务咨询"
    elif any(word in question_lower for word in ["价格", "多少钱", "费用"]):
        return "价格咨询"
    elif any(word in question_lower for word in ["预约", "时间", "安排"]):
        return "预约咨询"
    elif any(word in question_lower for word in ["地址", "位置", "怎么去"]):
        return "位置咨询"
    elif any(word in question_lower for word in ["优惠", "活动", "折扣"]):
        return "优惠咨询"
    else:
        return "其他咨询"

def train_model_with_samples(samples: List[Sample], task_type: str) -> bool:
    """使用样本训练模型"""
    global customer_service_adapter, intent_adapter
    
    if len(samples) < MIN_TRAINING_SAMPLES:
        print(f"样本数量不足，需要至少{MIN_TRAINING_SAMPLES}个样本，当前只有{len(samples)}个")
        return False
    
    try:
        if task_type == "customer_service":
            environment = KouQiangEnvironment("customer_service")
            print(f"开始训练客服模型，样本数量: {len(samples)}")
            customer_service_adapter.run(samples, environment, epochs=1)
            
            # 保存更新后的playbook
            customer_service_adapter.playbook.save_to_file(str(CUSTOMER_SERVICE_PLAYBOOK))
            print(f"客服playbook已更新并保存到: {CUSTOMER_SERVICE_PLAYBOOK}")
            return True
            
        elif task_type == "intent_classification":
            environment = KouQiangEnvironment("intent_classification")
            print(f"开始训练意图识别模型，样本数量: {len(samples)}")
            intent_adapter.run(samples, environment, epochs=1)
            
            # 保存更新后的playbook
            intent_adapter.playbook.save_to_file(str(INTENT_PLAYBOOK))
            print(f"意图识别playbook已更新并保存到: {INTENT_PLAYBOOK}")
            return True
            
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        return False
    
    return False

# API端点
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化适配器"""
    initialize_adapters()

@app.get("/")
async def root():
    """根端点，返回API信息"""
    return {
        "message": "口腔客服ACE训练API服务",
        "version": "1.0.0",
        "endpoints": {
            "/": "API信息",
            "/health": "健康检查",
            "/train": "训练模型",
            "/playbook/{task_type}": "获取playbook",
            "/generate/{task_type}": "生成回复"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "customer_service_adapter": customer_service_adapter is not None,
        "intent_adapter": intent_adapter is not None
    }

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """训练模型端点"""
    try:
        # 将对话数据转换为训练样本
        samples = process_dialogues_to_samples(request.dialogues, request.task_type)
        
        if len(samples) < MIN_TRAINING_SAMPLES:
            return TrainingResponse(
                success=False,
                message=f"样本数量不足，需要至少{MIN_TRAINING_SAMPLES}个样本，当前只有{len(samples)}个"
            )
        
        # 在后台任务中训练模型
        def train_in_background():
            success = train_model_with_samples(samples, request.task_type)
            print(f"后台训练完成，结果: {success}")
        
        background_tasks.add_task(train_in_background)
        
        playbook_path = None
        if request.task_type == "customer_service":
            playbook_path = str(CUSTOMER_SERVICE_PLAYBOOK)
        elif request.task_type == "intent_classification":
            playbook_path = str(INTENT_PLAYBOOK)
        
        return TrainingResponse(
            success=True,
            message=f"已接收{len(samples)}个样本，正在后台训练{request.task_type}模型",
            playbook_updated=True,
            playbook_path=playbook_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练请求处理失败: {str(e)}")

@app.get("/playbook/{task_type}", response_model=PlaybookResponse)
async def get_playbook(task_type: str):
    """获取playbook端点"""
    try:
        if task_type == "customer_service":
            if not customer_service_adapter:
                raise HTTPException(status_code=404, detail="客服适配器未初始化")
            
            playbook_data = customer_service_adapter.playbook.to_dict()
            return PlaybookResponse(
                success=True,
                playbook=playbook_data,
                message="成功获取客服playbook"
            )
            
        elif task_type == "intent_classification":
            if not intent_adapter:
                raise HTTPException(status_code=404, detail="意图识别适配器未初始化")
            
            playbook_data = intent_adapter.playbook.to_dict()
            return PlaybookResponse(
                success=True,
                playbook=playbook_data,
                message="成功获取意图识别playbook"
            )
            
        else:
            raise HTTPException(status_code=400, detail="无效的任务类型，必须是customer_service或intent_classification")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取playbook失败: {str(e)}")

@app.post("/generate/{task_type}")
async def generate_response(task_type: str, question: str, context: str = ""):
    """生成回复端点"""
    try:
        if task_type == "customer_service":
            if not customer_service_adapter:
                raise HTTPException(status_code=404, detail="客服适配器未初始化")
            
            sample = Sample(question=question, context=context)
            result = customer_service_adapter.generator.generate(
                question=sample.question,
                context=sample.context,
                playbook=customer_service_adapter.playbook,
                reflection="",
                sample=sample
            )
            
            return {
                "success": True,
                "question": question,
                "response": result.final_answer,
                "reasoning": result.reasoning,
                "bullet_ids": result.bullet_ids
            }
            
        elif task_type == "intent_classification":
            if not intent_adapter:
                raise HTTPException(status_code=404, detail="意图识别适配器未初始化")
            
            sample = Sample(question=question, context=context)
            result = intent_adapter.generator.generate(
                question=sample.question,
                context=sample.context,
                playbook=intent_adapter.playbook,
                reflection="",
                sample=sample
            )
            
            return {
                "success": True,
                "question": question,
                "intent": result.final_answer,
                "reasoning": result.reasoning,
                "bullet_ids": result.bullet_ids
            }
            
        else:
            raise HTTPException(status_code=400, detail="无效的任务类型，必须是customer_service或intent_classification")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回复失败: {str(e)}")

# 启动服务器
if __name__ == "__main__":
    # 确保数据目录存在
    DATA_DIR.mkdir(exist_ok=True)
    PLAYBOOK_DIR.mkdir(exist_ok=True)
    
    print(f"启动口腔客服ACE训练API服务...")
    print(f"服务地址: http://{API_HOST}:{API_PORT}")
    print(f"API文档: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)