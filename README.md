# 知伴 (ZhiBan) — AI 论文伴读系统

[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-blue)](https://github.com/IDWGU/zhiban-webui)
[![Python](https://img.shields.io/badge/python-3.12%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

知伴是一站式 AI 论文伴读工具，支持上传 PDF/DOCX/TXT/Markdown 论文，通过 **DeepSeek API** 进行 RAG 智能问答、论文翻译、向量检索。也支持本地模型作为离线备选。

**WebUI 一键版**：下载解压 → 双击 → 配置 DeepSeek API Key → 打开浏览器即用。

---

## 功能

- **智能问答** — 基于论文内容的 RAG 检索增强生成，Agent 自主多轮搜索
- **论文翻译** — 腾讯混元 Hy-MT2 翻译模型，PDF 逐句翻译
- **论文管理** — 导入 PDF/DOCX/TXT/Markdown，自动向量化建库
- **知识图谱** — 论文间引用关系网络，关联推荐
- **DeepSeek 驱动** — 接入 DeepSeek API，享受最新模型能力
- **本地备选** — 支持 llama.cpp / MLX 本地推理，离线可用

---

## 快速开始

### 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | macOS 14+ (Sonoma) |
| 芯片 | Apple Silicon (M1/M2/M3/M4) |
| 内存 | ≥8GB |
| 存储 | ≥5GB 可用空间 |
| Python | 3.12+（脚本自动检测，缺失时提示安装） |

### 启动步骤

1. **下载** — 从 [GitHub Releases](https://github.com/IDWGU/zhiban-webui/releases) 下载最新 `zhiban-webui-v*.zip`
2. **解压** — 双击解压到任意目录
3. **启动** — 双击 `启动知伴.command`
4. **配置** — 在设置页面填入 DeepSeek API Key（从 [DeepSeek 开放平台](https://platform.deepseek.com/) 获取）
5. **使用** — 浏览器自动打开 `http://localhost:18921`，导入论文即可开始对话

> **注**：首次启动如遇 macOS 安全提示，请右键 `启动知伴.command` → 打开

### DeepSeek API 配置

在知伴的设置面板中配置：

| 设置项 | 值 |
|--------|-----|
| API 地址 | `https://api.deepseek.com` |
| API Key | 你的 DeepSeek API Key |
| 模型 | `deepseek-chat` / `deepseek-reasoner` |

支持 DeepSeek 全系列模型，包括 reasoning 模式（`deepseek-reasoner`）。

---

## 项目结构

```
zhiban-webui/
├── README.md                    # 本文件
├── 启动知伴.command             # macOS 双击启动器
├── start-zhiban.sh              # 核心启动脚本（环境检测+模型下载+启动）
├── requirements.txt             # Python 依赖列表
├── sidecar/                     # Python 后端源码
│   ├── webui_launcher.py        # WebUI 启动入口
│   ├── server.py                # FastAPI + WebSocket 服务器
│   ├── config.py                # 全局配置
│   ├── engine/                  # 工作流引擎（分类+回答+Agent）
│   ├── agent/                   # Agent Loop 自主工具调用
│   ├── llm/                     # LLM 推理（API 优先 + 本地备选）
│   ├── rag/                     # RAG 检索（向量+知识图谱）
│   ├── translation/             # 论文翻译引擎
│   ├── handlers/                # WebSocket 消息处理
│   └── persistence.py           # SQLite 会话持久化
├── src/                         # React 前端源码
├── scripts/
│   ├── serve.py                 # 生产服务器入口
│   └── download-models.sh       # 模型下载（国内镜像轮换）
├── config/
│   └── mirrors.json             # 镜像源配置
└── models/                      # 本地模型存放（可选）
    ├── llm/                     # 对话模型 (*.gguf)
    ├── translation/             # 翻译模型 (*.gguf)
    └── embedding/               # 向量嵌入模型
```

---

## 使用本地模型（可选）

如果需要在离线环境使用，可选择下载本地模型：

### 对话模型

首次启动时可选择下载：

| 模型 | 大小 | 说明 |
|------|------|------|
| Qwopus3.5-9B-v3 (Q4_K_M) | ~5.4GB | 推荐，最佳对话质量 |
| Qwopus3.5-4B-v3 (Q4_K_M) | ~2.6GB | 轻量，适合低内存设备 |

### 翻译 & 嵌入模型（自动下载）

| 模型 | 大小 | 说明 |
|------|------|------|
| Hy-MT2-1.8B (Q4_K_M) | ~1.1GB | 腾讯混元翻译模型 |
| Jina Embedding v5 Nano | ~0.5GB | 向量嵌入模型 |

下载使用国内镜像自动轮换：hf-mirror.com / huggingface.modelscope.cn / aliendao.cn，单镜像 15s 超时自动切换，支持断点续传。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 18 + TypeScript 5.9 + Zustand 4 + Vite 6 |
| 后端 | Python 3.14 + FastAPI + WebSocket |
| 主 LLM | DeepSeek API（deepseek-chat / deepseek-reasoner） |
| 本地 LLM | llama-cpp-python / MLX (Apple Silicon) |
| 向量检索 | ChromaDB + Jina Embeddings v5 |
| 知识图谱 | NetworkX |
| 翻译 | 腾讯混元 Hy-MT2-1.8B |

---

## 常见问题

**Q: 必须用 DeepSeek API 吗？**
A: 推荐使用 DeepSeek API 获得最佳体验，但也支持本地模型离线使用。

**Q: 如何获取 DeepSeek API Key？**
A: 访问 [platform.deepseek.com](https://platform.deepseek.com/) 注册并创建 API Key，费用极低（约 ¥1/百万 token）。

**Q: 启动后浏览器显示 "无法连接"？**
A: 首次启动可能需要下载嵌入模型，请等待终端窗口显示 "知伴已启动" 后再刷新。

**Q: 提示 "Python 3.12+ 未找到"？**
A: 终端执行 `brew install python@3.14`，然后重新双击启动。

**Q: 如何更新？**
A: 下载新版本 zip 覆盖解压即可，模型文件和用户数据不会被覆盖。

**Q: 数据存储在哪里？**
A: 所有数据（对话记录、论文库、向量索引）存储在项目目录内，删除目录即彻底清除。

---

## 开发

```bash
# 安装依赖
npm install

# 开发模式
npm run dev

# 构建 Web 前端
npm run build:web

# 构建 Electron 桌面版
npm run build:dist
```
