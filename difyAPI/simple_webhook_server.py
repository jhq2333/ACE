#!/usr/bin/env python3
"""
简单的webhook服务器，用于测试
"""
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
import asyncio
import json
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_webhook")

# 创建FastAPI应用
app = FastAPI(title="Simple Webhook Server", description="用于测试的简单webhook服务器")

# 配置
PERSIST_PATH = os.path.join(os.path.dirname(__file__), "..", "dify_incoming.jsonl")
PERSIST_PATH = os.path.abspath(PERSIST_PATH)
EXPECTED_API_KEY = os.environ.get("DIFY_WEBHOOK_API_KEY", "changeme")

# 数据模型
class Message(BaseModel):
    role: str
    text: str

class WebhookPayload(BaseModel):
    conversation_id: Optional[str] = None
    user: Optional[str] = None
    messages: Optional[List[Message]] = None

# 内存队列
queue: asyncio.Queue = asyncio.Queue()

# 后台工作进程
async def persist_worker():
    """后台工作进程，将队列中的数据持久化到文件"""
    os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
    logger.info(f"持久化工作进程已启动，写入到 {PERSIST_PATH}")
    
    while True:
        item = await queue.get()
        try:
            with open(PERSIST_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info("已持久化数据项")
        except Exception as e:
            logger.error(f"持久化数据项失败: {str(e)}")
        finally:
            queue.task_done()

# 启动事件
@app.on_event("startup")
async def startup_event():
    """启动后台工作进程"""
    app.state.persist_task = asyncio.create_task(persist_worker())
    logger.info("简单webhook服务器已启动")

# API密钥验证
def verify_api_key(header_value: Optional[str]):
    """验证API密钥"""
    if header_value != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {"message": "Simple Webhook Server", "status": "running"}

# Webhook端点
@app.post("/webhook")
async def webhook(payload: WebhookPayload, x_dify_signature: Optional[str] = Header(None)):
    """接收webhook数据"""
    # 验证API密钥
    verify_api_key(x_dify_signature)
    
    # 处理数据
    data = payload.dict()
    incoming_items = []
    
    if isinstance(data.get("messages"), list):
        for m in data["messages"]:
            item = {
                "conversation_id": data.get("conversation_id"),
                "user": data.get("user"),
                "message": m.dict() if isinstance(m, BaseModel) else m,
            }
            incoming_items.append(item)
    else:
        incoming_items.append(data)
    
    # 将数据加入队列
    for item in incoming_items:
        await queue.put(item)
    
    logger.info(f"接收到 {len(incoming_items)} 条数据")
    return {"status": "accepted", "count": len(incoming_items)}

# 流式端点
@app.post("/stream")
async def stream_endpoint(request: Request, x_dify_signature: Optional[str] = Header(None)):
    """接收流式数据"""
    # 验证API密钥
    verify_api_key(x_dify_signature)
    
    # 处理流式数据
    async for chunk in request.stream():
        if not chunk:
            continue
        
        try:
            text = chunk.decode("utf-8")
        except Exception:
            continue
        
        for line in text.splitlines():
            if not line:
                continue
            
            if line.startswith("data:"):
                payload_text = line[len("data:"):].strip()
            else:
                payload_text = line.strip()
            
            if not payload_text:
                continue
            
            try:
                obj = json.loads(payload_text)
                await queue.put({"stream": True, "message": obj})
            except json.JSONDecodeError:
                logger.warning(f"收到非JSON流数据: {payload_text[:200]}")
    
    return {"status": "stream_received"}

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)