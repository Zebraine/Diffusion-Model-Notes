# 图解扩散模型 · 中文翻译

《The Annotated Diffusion Model》的中文全文翻译。页面是静态 HTML，已迁移到论文库的共享排版系统。

## 打开方式

从论文库首页进入：

```bash
xdg-open ../index.html
```

也可以直接打开本目录的 `index.html`。

## 目录结构

```text
annotated-diffusion/
├── index.html            # 完整文章页面
├── paper.json            # 论文库元数据（标题、作者、标签、封面、统计）
├── README.md
└── images/               # 文章配图（WebP/PNG/JPEG/GIF，均为本地文件）
```

共享资源在上一级：

```text
../assets/style.css       # 设计令牌、页头页脚、论文版式
../assets/article.js      # 目录高亮、复制代码、图片灯箱、阅读进度
../assets/pygments.css    # 代码高亮配色
../assets/vendor/katex/   # 本地 KaTeX 样式与字体（公式在构建时预渲染）
```

## 页面内容

- 10 个章节，包含从 DDPM 直觉、数学推导到 PyTorch 完整实现的全部正文。
- 11 幅文章配图，带中文图注、来源标注与点击放大。
- 29 个 Python 代码块，支持一键复制；注释已译为中文。
- 125 处公式使用 KaTeX 预渲染，保留 MathML 语义。
- 桌面端含粘性目录、阅读进度与滚动高亮；移动端使用可折叠目录。
- 适配 320px–1440px 以上宽度，并提供打印样式。

## 来源与版权

- 原文：[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- 作者：Niels Rogge、Kashif Rasul（Hugging Face）
- 原发布日期：2022 年 6 月 7 日
- 原始代码：[huggingface/blog](https://github.com/huggingface/blog/blob/main/annotated-diffusion.md)、[Colab Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb)

本页面为学习交流用途的中文翻译。原文、插图及相关材料的版权归 Hugging Face 与原作者所有。KaTeX 依据 MIT License 使用，许可证见 `../assets/vendor/katex/LICENSE`。
