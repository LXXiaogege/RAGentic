# RAGentic Docker 部署指南

## 快速开始

### 1. 基础部署（仅 Web 应用）

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 访问应用
# http://localhost:7860
```

### 2. 带 Redis 缓存部署

```bash
# 启动 Web + Redis
docker-compose --profile with-redis up -d
```

### 3. 带 Milvus 生产环境部署

```bash
# 启动 Web + Milvus
docker-compose --profile with-milvus up -d

# 启动所有服务
docker-compose --profile with-redis --profile with-milvus up -d
```

## 环境变量配置

复制 `.env.example` 为 `.env` 并配置：

```bash
# LLM 配置
LLM__API_KEY=your_api_key
LLM__BASE_URL=https://api.example.com/v1

# Embedding 配置
EMBEDDING__API_KEY=your_api_key
EMBEDDING__BASE_URL=https://api.example.com/v1

# Langfuse 配置（可选）
LANGFUSE__HOST=https://cloud.langfuse.com
LANGFUSE__PUBLIC_KEY=your_public_key
LANGFUSE__SECRET_KEY=your_secret_key
```

## 数据持久化

数据存储在以下目录：
- `./data/` - 知识库、数据库文件
- `./logs/` - 日志文件

确保这些目录在主机上有足够空间。

## 健康检查

```bash
# 检查容器健康状态
docker-compose ps

# 手动健康检查
curl http://localhost:7860/api/health
```

## 停止服务

```bash
# 停止所有服务
docker-compose down

# 停止并删除数据卷（谨慎使用）
docker-compose down -v
```

## 更新部署

```bash
# 重新构建并启动
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 故障排查

### 查看日志
```bash
docker-compose logs -f web
```

### 进入容器调试
```bash
docker-compose exec web bash
```

### 重启服务
```bash
docker-compose restart web
```
