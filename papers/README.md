# 论文库 / Paper Library

入口：打开 [`index.html`](index.html)。整个目录不依赖网络，图片、字体、样式和脚本都在本地。

## 目录结构

```text
papers/
├── index.html                  # 论文库首页（由 tools/build.mjs 生成）
├── assets/                     # 所有论文共享的排版系统
│   ├── style.css               # 设计令牌、页头页脚、论文版式、表格、算法、图注
│   ├── home.css                # 首页 hero、论文卡片、搜索与标签筛选
│   ├── article.js              # 目录高亮、阅读进度、复制代码、图片灯箱
│   ├── home.js                 # 首页搜索 / 标签筛选
│   ├── pygments.css            # Pygments GitHub Dark 代码配色（生成物）
│   └── vendor/katex/           # KaTeX 0.16.11 样式与字体
├── annotated-diffusion/        # 001 · 静态页面，元数据在 paper.json
│   ├── index.html
│   ├── paper.json
│   └── images/
└── ddpm/                       # 002 · 由 content.md 构建
    ├── content.md
    ├── index.html              # 生成物，不要手改
    ├── paper.json
    └── images/
```

## 新增一篇论文

### 1. 建目录与元数据

```bash
mkdir -p papers/<slug>/images
```

新建 `papers/<slug>/paper.json`：

```json
{
  "slug": "ddpm",
  "order": 2,
  "kind": "Paper 精译",
  "series": "NeurIPS 2020",
  "title": "去噪扩散概率模型",
  "title_lead": "去噪",
  "title_accent": "扩散概率模型",
  "title_en": "Denoising Diffusion Probabilistic Models",
  "dek": "一句话导语，显示在 hero 与首页卡片。",
  "authors": [
    { "name": "Jonathan Ho", "url": "https://arxiv.org/abs/2006.11239" }
  ],
  "date": "2020-06-19",
  "year": "2020",
  "venue": "NeurIPS 2020",
  "reading_minutes": 45,
  "tags": ["扩散模型基础", "数学推导"],
  "summary": "首页卡片摘要。",
  "cover": "images/cover.webp",
  "cover_alt": "封面图的替代文本",
  "cover_caption": "封面图短说明",
  "cover_meta": "生成样本 · DDPM · 2020",
  "cover_ratio": "16x9",
  "cover_fit": "cover",
  "source_url": "https://arxiv.org/abs/2006.11239",
  "pdf_url": "https://arxiv.org/pdf/2006.11239",
  "code_url": "https://github.com/hojonathanho/diffusion",
  "disclaimer": "版权与翻译说明。",
  "build": true
}
```

`order` 决定编号、首页排序与上一篇/下一篇。`cover` 相对于论文目录。`cover_ratio` 支持 `3x1`、`16x9`、`2x1`，默认 `3x1`；`cover_fit` 可选 `cover`（默认，裁切）或 `contain`（完整显示，适合宽幅三联图）。

### 2. 写正文

新建 `papers/<slug>/content.md`。支持的约定：

- 标题：`## 引言 {#introduction}`、`### 3.1 前向过程与 L_T {#forward-process}`；`{#id}` 会变成锚点。
- 行内公式：`$q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$`。
- 行间公式：`$$ ... \tag{1} $$`；`\tag{1}` 会渲染成右侧编号，且公式区域可横向滚动。
- 代码块：用三个反引号加语言名（如 `python`），构建时由 Pygments 高亮，自动带复制按钮。
- 图片 / 表格 / 算法：直接写 HTML，复用共享类名。完整示例见 `ddpm/content.md`。
  - 图片：`figure.article-figure` + `a.zoom-link` + `figcaption` + `span.fig-no`
  - 表格：`div.table-wrap` + `table.paper-table`
  - 算法：`div.algorithm` + `div.algorithm-head` + `div.algorithm-body`
  - 摘要：`div.abstract-card`
- 外部链接会自动加 `target="_blank"` 与 `rel="noopener noreferrer"`。

### 3. 构建

```bash
cd tools
npm install        # 首次需要；安装 markdown-it 与 katex
node build.mjs     # 生成论文页 + 首页
node build.mjs --check   # 只检查本地引用与页面是否存在
```

构建会：

- 把 `content.md` 编译成 `papers/<slug>/index.html`（KaTeX 预渲染、Pygments 高亮、自动目录）。
- 根据所有 `paper.json` 重新生成 `papers/index.html`（卡片、统计、标签筛选、上一篇/下一篇）。
- 校验图片、样式、脚本等本地引用是否存在。

### 4. 只想写静态页？

在 `paper.json` 里去掉 `"build": true`，直接提供 `papers/<slug>/index.html`，并引用 `../assets/style.css` 与 `../assets/article.js`。首页仍会自动收录它。现有的 `annotated-diffusion/` 就是这种模式。

## 设计约定

- 无 CSS 框架：`style.css` 是手写设计系统，令牌在 `:root`（`--paper`、`--ink`、`--accent`、`--amber` 等）。
- 两类页面共享同一套页头、页脚、按钮与字体；首页额外加载 `home.css`。
- 公式在构建时预渲染，运行时不加载 KaTeX JS；代码高亮的配色来自 Pygments 的 `github-dark`。
- 目标宽度 320px–1440px+，并包含 `prefers-reduced-motion` 与打印样式。

## 依赖

- Node.js 18+（构建脚本，ESM）
- Python 3 + Pygments（代码高亮）
- 不需要前端框架、数据库或网络服务。

## 版权

各篇论文的原文、图表与代码版权归原作者所有；本库仅作学习交流用途。每篇页面底部有对应的来源与版权说明。
