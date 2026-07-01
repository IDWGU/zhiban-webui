# 知伴论文翻译功能 — 产品需求文档 (PRD)

## Problem Statement

知伴用户目前可以拖入英文PDF论文进行阅读和AI伴读问答，但无法对论文正文进行全文翻译。用户（研究生/科研人员）阅读英文论文时，常需要快速获取中文译文来辅助理解，但又不想切换到外部翻译工具（DeepL/Google Translate等）打断阅读流。

现有外部翻译工具的痛点：
- 复制粘贴段落翻译，割裂阅读体验
- PDF上传翻译后板式混乱（PDF固定坐标 vs 译文长度变化）
- 无法保留原文结构（标题层级、表格、公式、引用标记）
- 翻译结果不可编辑，无法做笔记或修正术语

## Solution

在知伴内增加**一键论文翻译**功能，将当前打开的论文翻译为中文，输出为一个**结构保留的 Word (.docx) 文档**，用户可在Word中继续阅读、编辑、标注。

为什么是 Word 而非 PDF回填：
- Word天然处理流式排版，译文长度变化（中英互译可能30%-200%长度差异）由Word自动reflow，零溢出风险
- 表格、公式、图片标题等复杂元素由python-docx结构化管理，不需要像素级坐标计算
- 用户可以继续编辑译文、修正术语、添加笔记
- 实现复杂度显著低于PDF文字回填（无需版面分析坐标、字体嵌入、溢出处理）
- 通过 LibreOffice headless / Pandoc 也可后续导出为PDF

核心流程：`PyMuPDF提取结构 → LLM分段翻译 → python-docx生成Word → Electron保存对话框`

## User Stories

1. 作为研究生，我希望点击一个按钮就能将整篇英文PDF论文翻译成中文Word文档，这样我可以在Word中快速浏览全文中文内容
2. 作为研究生，我希望翻译结果保留原文的标题层级（一级标题、二级标题等），这样我能快速定位到想看的章节
3. 作为研究生，我希望翻译结果保留论文中的表格（翻译表头和数据），这样数据内容不会丢失
4. 作为研究生，我希望翻译结果保留原文中的引用标记（如 [1]、[23]），这样我能对应到参考文献
5. 作为研究生，我希望数学公式在翻译结果中原样保留，不被翻译破坏
6. 作为研究生，我希望图片原样保留在翻译文档中，但图题被翻译成中文
7. 作为研究生，我希望看到翻译进度（流式输出），而不是等几分钟后一次性拿到结果
8. 作为研究生，我希望翻译过程利用我已经配置好的LLM API Key，不需要额外付费订阅翻译服务
9. 作为研究生，我希望翻译结果是一个可编辑的Word文档，如果某个术语翻译不准确我可以直接修改
10. 作为研究生，我希望可以选择翻译范围：全文 / 当前章节 / 选中的段落
11. 作为研究生，我希望翻译时能指定领域术语表（如电化学领域的专有名词），确保术语翻译一致性
12. 作为研究生，我希望导出时可以选择布局格式：纯译文 / 双语对照（左原文右译文）
13. 作为一个只懂中文的用户，我希望论文的图表标题都翻译了，这样不需要对照原文看图

## Implementation Decisions

### 架构决策

**方案：Python sidecar 内新增翻译管道，不引入额外服务**

知伴已有 Python FastAPI sidecar（PyMuPDF已安装，LLM代理已存在，WebSocket流式已打通），翻译管道全部在sidecar内实现。前端仅增加触发按钮和进度展示，翻译引擎和文档生成为纯后端逻辑。

**不选方案**：
- 纯前端翻译：需在浏览器内运行pdf.js提取+翻译+生成docx，pdf.js文本提取不如PyMuPDF精确，且docx生成在浏览器端文件体积大、字体嵌入困难
- 独立翻译微服务：引入运维复杂度，与知伴桌面端定位不符

### 新增模块

| 模块 | 位置 | 职责 | 接口 |
|------|------|------|------|
| **TranslationPipeline** | `sidecar/translation/pipeline.py` | 编排翻译全流程：提取→切分→翻译→生成 | `async translate_pdf(pdf_path, options) -> docx_path` |
| **DocExtractor** | `sidecar/translation/extractor.py` | PDF/DOCX结构化提取：段落、标题、表格、公式、图片、引用 | `extract(pdf_path) -> StructuredDocument` |
| **DocxGenerator** | `sidecar/translation/docx_gen.py` | 将StructuredDocument + 翻译结果生成Word | `generate(structured_doc, translations, options) -> bytes` |
| **TranslationHandler** | `sidecar/translation/handler.py` | WebSocket消息处理：接收翻译请求、流式推送进度、完成后通知 | WS消息协议 |
| **TranslationPanel** (前端) | `src/components/translation/` | 翻译触发按钮、进度条、选项面板 | React组件 |

### 关键类型定义

```python
# StructuredDocument — 结构化文档模型
@dataclass
class StructuredBlock:
    type: Literal["heading", "paragraph", "table", "figure", "formula", "reference"]
    level: int | None           # heading level 1-6
    text: str
    translation: str | None     # filled after translation
    children: list[StructuredBlock]  # for table cells etc.
    metadata: dict               # e.g. {"ref_id": "[1]", "img_path": "..."}

class StructuredDocument:
    title: str
    blocks: list[StructuredBlock]
    metadata: dict  # author, page_count, etc.
```

```typescript
// 前端 — 扩展 WS 消息协议
interface TranslationRequestMessage {
  type: 'translation_request'
  filePath: string
  options: {
    scope: 'full' | 'chapter' | 'selection'
    layout: 'translated_only' | 'bilingual'
    glossary?: Record<string, string>   // 术语表
    targetLang: string                   // 默认 'zh-CN'
  }
  apiKey?: string
  model?: string
}

interface TranslationProgressMessage {
  type: 'translation_progress'
  phase: 'extracting' | 'translating' | 'generating'
  current: number
  total: number
  currentBlockText?: string     // 当前正在翻译的原文
  translatedBlockText?: string  // 已翻译的文本（流式）
}

interface TranslationDoneMessage {
  type: 'translation_done'
  docxPath: string
  totalBlocks: number
  duration: number
}
```

### 翻译策略细化

| 元素类型 | 翻译策略 | 上下文传递 |
|---------|---------|-----------|
| 标题 (h1-h6) | LLM翻译，保留编号如 "1. Introduction" → "1. 引言" | 同级标题作为上下文 |
| 正文段落 | LLM翻译，保留引用标记 `[1]`、`[23]` | 前后各1段作为上下文 |
| 表格 | 逐单元格LLM翻译，保持表格结构不变 | 同列表头 + 同行上下文 |
| 图片 | 原文原样搬运，仅翻译图题 `<figcaption>` | 无 |
| 公式 | **不翻译**，原样保留 | 无 |
| 参考文献 | **不翻译**，保留原文 | 无 |
| 页眉页脚 | 可选翻译 | 无 |

### 分段策略与LLM调用优化

为避免长文档翻译时token消耗过大和上下文窗口溢出：

1. 按章节分段，每章作为一个翻译批次
2. 每个批次：将当前章节的所有段落 + 前一章最后一段（上下文锚点）一起发送
3. LLM提示词要求：逐段翻译，段落间用 `---PARA_BREAK---` 分隔，保留引用标记和LaTeX公式
4. 后端解析LLM输出，按分隔符切分回填到StructuredBlock
5. 超长段落（>2000字符）单独翻译

### 流式进度方案

翻译进度通过现有WebSocket推送：
- Phase 1 `extracting`：正在提取文档结构（通常1-3秒）
- Phase 2 `translating`：每完成一个章节翻译，推送 `{current, total}`，并附上刚翻译完成的段落预览
- Phase 3 `generating`：正在生成Word文档（通常2-5秒）
- 完成后推送 `translation_done`，包含生成的docx文件路径

前端在PaperViewer工具栏显示翻译进度条，完成后弹出保存对话框（调用Electron dialog.showSaveDialog）。

### 术语表机制

- 内置一套通用学术术语表（如 "Abstract→摘要", "Introduction→引言", "Conclusion→结论"）
- 用户可在设置面板自定义领域术语表（JSON格式）
- 翻译时将术语表注入LLM system prompt：`"请使用以下术语翻译：Abstract → 摘要, electrode → 电极, ..."`
- 可考虑后续支持从知识库（brain/）中的中英文论文对自动提取术语

### 依赖变更

**新增Python依赖** (`sidecar/requirements.txt`)：
```
python-docx>=1.1.0        # Word文档生成
```
PyMuPDF (fitz) 已存在，无需新增。如需更精确的表格检测，可后续加入 DocLayout-YOLO。

**无前端npm依赖变更**。

### 设置面板扩展

在现有 SettingsPanel 的 LLM 标签页中新增：
- 翻译术语表编辑区（JSON textarea）
- 默认输出布局选项（纯译文 / 双语对照）
- 翻译目标语言选择（默认中文）

## Testing Decisions

### 测试原则
- 测试行为而非实现细节：测试"给定一篇PDF，翻译后docx包含全部段落"而非测试内部数据结构
- 优先测试边界情况：空文档、纯公式文档、超大表格、无标题文档

### 被测模块与测试策略

| 模块 | 测试类型 | 测试内容 |
|------|---------|---------|
| DocExtractor | 单元测试 | 准备一个小型手工PDF（含标题/段落/表格/公式），验证提取的StructuredDocument结构完整 |
| TranslationPipeline | 集成测试 | 用Mock LLM（固定返回译文），验证端到端流程：PDF入→docx出 |
| DocxGenerator | 单元测试 | 给定固定StructuredDocument，验证生成的docx包含正确的段落数和样式 |
| WS Handler | 集成测试 | 模拟前端发送translation_request，验证进度消息时序正确 |
| TranslationPanel | 组件测试 | 验证各状态（空闲/提取中/翻译中/生成中/完成/错误）的UI表现 |

### 测试数据
- 使用项目 `brain/` 目录中的论文全文（.txt）作为真实测试数据
- 构造特殊测试PDF：空文档、纯数学公式文档、20列表格文档

### 手工验证清单
- [ ] 翻译一篇标准学术论文（含Abstract/Introduction/Methods/Results/Conclusion）
- [ ] 翻译一篇含复杂表格（多级表头、合并单元格）的论文
- [ ] 翻译一篇含大量数学公式的论文，验证公式不被破坏
- [ ] 验证引用标记（[1], [2]等）在翻译后保留
- [ ] 验证翻译后Word文档在Word/WPS/Pages中打开正常
- [ ] 验证双语对照模式：左栏原文右栏译文，段落对齐
- [ ] 验证暗色/亮色主题下翻译面板UI正常

## Out of Scope

- **PDF扫描件翻译**：本功能仅处理可编辑PDF（含文本层）。扫描件OCR翻译需要Vision LLM API（如GPT-4V/Claude Vision），成本和技术复杂度差异大，作为后续迭代项
- **翻译后回填PDF保持版式**：正如前文讨论，PDF固定坐标回填在文本长度变化场景下效果差、实现复杂，不纳入本期范围
- **实时逐句翻译伴读（如豆包AI伴读模式）**：属于在线阅读器内嵌翻译覆盖层的方案，与本期"导出Word"是不同产品形态，后续可迭代
- **多语言翻译（除中英互译外）**：本期仅做英文→中文。LLM天然支持多语言，但UI和术语表仅针对中英优化
- **翻译记忆/翻译缓存**：不缓存历史翻译结果。同一篇论文重复翻译每次都重新调用LLM
- **端到端纯LLM图片翻译**：不采用文档截图→多模态模型→译文路径，使用传统的提取+翻译分离管道
- **PDF页面级版式分析（DocLayout-YOLO等）**：本期不做。PyMuPDF的block级提取已足够区分标题/正文/表格，后续如需更精确的期刊双栏、图文混排，再引入版面分析模型

## Further Notes

### 与现有功能的协同

1. **知识库检索**：翻译后的中文文本可自动索引到ChromaDB，增强中文查询的RAG检索效果
2. **论文笔记**：翻译产生的术语表可沉淀到NotesPanel，成为用户的个性化术语库
3. **LLM代理复用**：翻译直接使用现有 DeepSeek Proxy（`sidecar/llm/deepseek_proxy.py`），不需要新的LLM接入。翻译时关闭thinking模式以加速

### 性能预估

基于DeepSeek API典型速度（~50 tok/s）：
- 提取阶段：~2秒（PyMuPDF解析）
- 翻译阶段：一篇典型的8页论文约4000词 → 约6000 token输出 → ~120秒
- 生成阶段：~3秒（python-docx写入）
- **总计**：约2分钟/篇

优化方向：
- 并行翻译：不同章节可并发调用LLM API（DeepSeek支持多个并发请求）
- 流式生成：边翻译边写docx（而非等全部翻完再生成），提前让用户看到进度

### 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| LLM翻译质量不稳定（术语不一致、漏译） | 术语表注入prompt + 前后文上下文锚点 |
| 长文档超出LLM上下文窗口 | 按章节分段，每段独立翻译 |
| PDF中表格提取不完整（PyMuPDF对复杂表格支持有限） | 先用基础方案，后续迭代加入Camelot-py/Docling TableFormer |
| python-docx生成大docx性能问题 | 论文长度通常<50页，python-docx在此范围内表现良好 |
| DeepSeek API调用失败/超时 | 已有retry逻辑（deepseek_proxy.py），翻译失败时保留已生成部分 |

### 后续迭代方向

1. **AI伴读翻译模式**：在PaperViewer内嵌双语覆盖层，可实现豆包式的"看一段翻一段"
2. **PDF扫描件支持**：集成GOT-OCR 2.0或Zerox模式，纯LLM Vision翻译扫描件
3. **翻译记忆库**：缓存已翻译段落，同领域论文复用翻译结果，降低API成本
4. **术语自动提取**：从知识库的中英文论文对中自动构建领域术语表
