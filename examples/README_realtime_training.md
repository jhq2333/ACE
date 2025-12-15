# 口腔客服ACE实时训练系统

本系统实现了通过HTTP协议将Dify机器人的对话数据实时传入ACE训练系统，更新playbook后再传回给机器人的完整流程。

## 系统架构

```
Dify机器人 ←→ Dify-ACE集成服务 ←→ ACE训练API ←→ ACE模型
```

1. **Dify机器人**: 提供对话服务，生成回复
2. **Dify-ACE集成服务**: 接收Dify机器人的对话数据，缓存并定期触发训练
3. **ACE训练API**: 提供RESTful API接口，用于训练ACE模型和获取更新后的playbook
4. **ACE模型**: 基于自适应上下文引擎的智能回复生成模型

## 文件说明

- `kouqiang_ace_training_api.py`: ACE训练API服务，提供RESTful接口
- `dify_ace_integration.py`: Dify机器人与ACE训练API集成服务
- `kouqiang_ace_training.py`: 原始的ACE训练脚本（已修改为使用全局配置）

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn pydantic requests flask
```

### 2. 启动ACE训练API

```bash
python examples/kouqiang_ace_training_api.py
```

服务将在 http://localhost:8000 启动，API文档可通过 http://localhost:8000/docs 访问。

### 3. 启动Dify-ACE集成服务

```bash
python examples/dify_ace_integration.py --mode server
```

服务将在 http://localhost:5000 启动，提供以下端点：
- `/webhook`: 接收Dify机器人的webhook数据
- `/manual_train`: 手动触发训练
- `/get_playbook/<task_type>`: 获取指定类型的playbook
- `/generate`: 使用ACE生成回复

### 4. 配置Dify机器人

在Dify机器人配置中设置webhook URL为：
```
http://your-server:5000/webhook
```

## API接口说明

### ACE训练API接口

#### 1. 健康检查
```
GET /health
```

#### 2. 训练模型
```
POST /train
Content-Type: application/json

{
  "dialogues": [
    {
      "session_id": "session_1",
      "messages": [
        {"role": "visitor", "content": "用户问题"},
        {"role": "agent", "content": "机器人回复"}
      ]
    }
  ],
  "task_type": "customer_service"  // 或 "intent_classification"
}
```

#### 3. 获取playbook
```
GET /playbook/{task_type}
```

#### 4. 生成回复
```
POST /generate/{task_type}
Content-Type: application/x-www-form-urlencoded

question=用户问题&context=上下文
```

### Dify-ACE集成服务接口

#### 1. 接收webhook
```
POST /webhook
Content-Type: application/json

{
  "conversation_id": "session_1",
  "query": "用户问题",
  "answer": "机器人回复"
}
```

#### 2. 手动触发训练
```
POST /manual_train
Content-Type: application/json

{
  "task_type": "customer_service"  // 或 "intent_classification"
}
```

#### 3. 获取playbook
```
GET /get_playbook/{task_type}
```

#### 4. 生成回复
```
POST /generate
Content-Type: application/json

{
  "question": "用户问题",
  "task_type": "customer_service",  // 或 "intent_classification"
  "context": "上下文"
}
```

## 工作流程

1. **对话数据收集**: Dify机器人与用户对话后，通过webhook将对话数据发送到Dify-ACE集成服务
2. **数据缓存**: 集成服务将对话数据缓存起来，直到达到训练条件（如对话数量或时间间隔）
3. **触发训练**: 当满足训练条件时，集成服务向ACE训练API发送训练请求
4. **模型训练**: ACE训练API接收对话数据，转换为训练样本，并使用这些样本训练模型
5. **更新playbook**: 训练完成后，更新后的playbook被保存到文件系统
6. **获取更新**: Dify-ACE集成服务可以获取更新后的playbook
7. **生成回复**: 使用更新后的playbook，ACE可以生成更准确的回复

## 配置说明

### ACE训练API配置

在 `kouqiang_ace_training_api.py` 中可以配置以下参数：

```python
# API配置
API_HOST = "0.0.0.0"
API_PORT = 8000

# ACE训练配置
LLM_MODEL = "openai/qwen-max-latest"
API_KEY = "your_api_key"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEMPERATURE = 0.7
MAX_TOKENS = 500
TIMEOUT = 60

# 训练配置
BATCH_SIZE = 5  # 每批处理的对话数量
MIN_TRAINING_SAMPLES = 3  # 最小训练样本数

# 文件路径
CUSTOMER_SERVICE_PLAYBOOK = "kouqiang_customer_service_playbook.json"
INTENT_PLAYBOOK = "kouqiang_intent_playbook.json"
```

### Dify-ACE集成服务配置

在 `dify_ace_integration.py` 中可以配置以下参数：

```python
# ACE训练API配置
ACE_API_BASE_URL = "http://localhost:8000"

# Dify机器人配置（示例）
DIFY_API_BASE_URL = "https://api.dify.ai/v1"
DIFY_API_KEY = "your_dify_api_key_here"

# 缓存配置
CACHE_SIZE = 10  # 缓存的对话数量
TRAINING_INTERVAL = 3600  # 训练间隔（秒），1小时
```

## 示例使用

### 1. 手动添加对话并触发训练

```python
from examples.dify_ace_integration import DifyACEIntegration

# 创建集成实例
dify_ace = DifyACEIntegration()

# 添加对话
dify_ace.add_dialogue("session_1", "visitor", "你们诊所有什么牙齿美白项目？")
dify_ace.add_dialogue("session_1", "agent", "我们提供冷光美白和家庭美白两种方式...")

# 触发训练
training_result = dify_ace.trigger_training()
print(f"训练结果: {training_result}")

# 获取更新后的playbook
updated_playbook = dify_ace.get_updated_playbook()
print(f"更新后的playbook包含{len(updated_playbook.get('bullets', []))}条策略")
```

### 2. 使用ACE生成回复

```python
# 使用ACE生成回复
test_question = "你们诊所有什么牙齿美白项目？"
ace_response = dify_ace.generate_response_with_ace(test_question)
print(f"ACE生成的回复: {ace_response}")
```

## 注意事项

1. **API密钥安全**: 请确保API密钥的安全，不要将其提交到版本控制系统
2. **资源使用**: 训练过程会消耗计算资源，建议在服务器上运行，并监控资源使用情况
3. **数据质量**: 训练数据的质量直接影响模型效果，建议定期检查和清理训练数据
4. **训练频率**: 过于频繁的训练可能导致系统不稳定，建议根据实际需求调整训练间隔
5. **错误处理**: 系统已实现基本的错误处理，但建议在生产环境中添加更完善的监控和告警机制

## 扩展功能

1. **多任务支持**: 当前系统支持客服回复和意图识别两种任务，可以扩展更多任务类型
2. **A/B测试**: 可以实现A/B测试功能，比较不同模型版本的效果
3. **用户反馈**: 可以添加用户反馈机制，收集用户对回复质量的评价
4. **模型评估**: 可以添加模型评估功能，定期评估模型性能
5. **分布式训练**: 可以扩展为分布式训练，处理更大规模的数据

## 故障排除

1. **训练失败**: 检查API密钥是否正确，网络连接是否正常
2. **内存不足**: 减少BATCH_SIZE或增加系统内存
3. **回复质量差**: 检查训练数据质量，调整模型参数
4. **API响应慢**: 检查网络连接，考虑使用更快的API服务

## 联系方式

如有问题或建议，请联系开发团队。