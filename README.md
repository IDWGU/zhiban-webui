# 知伴 (ZhiBan) — AI 论文伴读系统

> **本项目由 DeepSeekV4Pro 生成**

[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-blue)](https://github.com/zhiban-webui/zhiban-webui)
[![Python](https://img.shields.io/badge/python-3.12%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

知伴是一站式 AI 论文伴读工具，支持上传 PDF/DOCX/TXT/Markdown 论文，通过本地 LLM 进行 RAG 智能问答、论文翻译、向量检索。

**WebUI 一键版**：下载解压 → 双击 → 自动检测环境 → 自动下载模型 → 打开浏览器即用。

---

## 功能

- **智能问答** — 基于论文内容的 RAG 检索增强生成，支持多轮对话
- **论文翻译** — 腾讯混元 Hy-MT2 翻译模型，PDF 逐句翻译
- **论文管理** — 导入 PDF/DOCX/TXT/Markdown，自动向量化建库
- **知识图谱** — 论文间引用关系网络，关联推荐
- **本地运行** — 所有模型本地推理，数据不外传
- **Agent 自主搜索** — AI 自主决定搜索策略，多轮检索

---

## 项目结构

```
zhiban-webui/
├── README.md                    # 本文件
├── 启动知伴.command             # macOS 双击启动器
├── start-zhiban.sh              # 核心启动脚本（环境检测+模型下载+启动）
├── requirements.txt             # Python 依赖列表
├── web/                         # React 前端静态文件
├── sidecar/                     # Python 后端源码
│   ├── webui_launcher.py        # WebUI 启动入口
│   ├── server.py                # FastAPI + WebSocket 服务器
│   ├── config.py                # 全局配置
│   ├── engine/                  # 工作流引擎（分类+回答+Agent）
│   ├── agent/                   # Agent Loop 自主工具调用
│   ├── llm/                     # LLM 推理（llama.cpp / MLX / API）
│   ├── rag/                     # RAG 检索（向量+知识图谱）
│   ├── translation/             # 论文翻译引擎
│   ├── handlers/                # WebSocket 消息处理
│   └── persistence.py           # SQLite 会话持久化
├── scripts/
│   ├── serve.py                 # 生产服务器入口
│   └── download-models.sh       # 模型下载（国内镜像轮换）
├── config/
│   └── mirrors.json             # 镜像源配置
├── models/                      # 模型存放（首次启动自动下载）
│   ├── llm/                     # 对话模型 (*.gguf)
│   ├── translation/             # 翻译模型 (*.gguf)
│   └── embedding/               # 向量嵌入模型
└── brain/                       # 知识库
    ├── paper-texts/             # 论文原文
    ├── paper-reading/           # 阅读笔记
    └── library/                 # 论文库
```

---

## 一键启动

### 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | macOS 14+ (Sonoma) |
| 芯片 | Apple Silicon (M1/M2/M3/M4) |
| 内存 | ≥16GB 推荐（≥8GB 可用轻量模型） |
| 存储 | ≥15GB 可用空间 |
| Python | 3.12+（脚本自动检测，缺失时提示安装） |

### 启动步骤

1. **下载** — 从 [GitHub Releases](https://github.com/zhiban-webui/zhiban-webui/releases) 下载最新 `zhiban-webui-v*.zip`
2. **解压** — 双击解压到任意目录
3. **启动** — 双击 `启动知伴.command`
4. **等待** — 首次启动会：
   - 检测 Python 环境（缺失则提示安装）
   - 安装 Python 依赖（清华镜像）
   - 下载所需模型（国内镜像，约 5-8GB）
5. **使用** — 浏览器自动打开 `http://localhost:18921`

> **注**：首次下载如遇 macOS 安全提示，请右键 `启动知伴.command` → 打开

### 模型下载

首次启动时可选择下载的对话模型：

| 模型 | 大小 | 说明 |
|------|------|------|
| Qwopus3.5-9B-v3 (Q4_K_M) | ~5.4GB | 推荐，最佳对话质量 |
| Qwopus3.5-4B-v3 (Q4_K_M) | ~2.6GB | 轻量，适合低内存设备 |

翻译模型和嵌入模型自动下载：

| 模型 | 大小 | 说明 |
|------|------|------|
| Hy-MT2-1.8B (Q4_K_M) | ~1.1GB | 腾讯混元翻译模型 |
| Jina Embedding v5 Nano | ~0.5GB | 向量嵌入模型 |

下载使用国内镜像（自动轮换）：
- hf-mirror.com
- huggingface.co
- huggingface.modelscope.cn
- aliendao.cn

单个镜像 15s 超时自动切换，支持断点续传。

### 使用自定义模型

如果你已有 GGUF 模型文件：
- 放入 `models/llm/` 目录，启动时自动发现
- 或设置环境变量 `ZHIBAN_MODEL_DIR` 指向你的模型目录

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 18 + TypeScript 5.9 + Zustand 4 + Vite 6 |
| 后端 | Python 3.14 + FastAPI + WebSocket |
| LLM 推理 | llama-cpp-python / MLX (Apple Silicon) |
| 向量检索 | ChromaDB + sentence-transformers |
| 知识图谱 | NetworkX |
| 翻译 | 腾讯混元 Hy-MT2-1.8B |

---

## 常见问题

**Q: 启动后浏览器显示 "无法连接"？**
A: 首次启动需要下载模型，请等待终端窗口显示 "知伴已启动" 后再刷新浏览器。

**Q: 模型下载失败？**
A: 脚本会自动切换 4 个镜像源重试。如全部失败，请检查网络或手动将 GGUF 文件放入对应 models/ 目录。

**Q: 提示 "Python 3.12+ 未找到"？**
A: 终端执行 `brew install python@3.14`，然后重新双击启动。

**Q: 如何更新？**
A: 下载新版本 zip 覆盖解压即可，模型文件不会被覆盖。

**Q: 数据存储在哪里？**
A: 所有数据（对话记录、论文库、向量索引）存储在项目目录内，删除目录即彻底清除。

---

## 开发

```bash
# 开发模式
npm install
npm run dev

# 构建 Web 前端
npm run build:web

# 构建 Electron 桌面版
npm run build:dist
```
