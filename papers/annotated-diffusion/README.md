# 图解扩散模型 · 中文翻译

《The Annotated Diffusion Model》的中文全文翻译，保存为可直接打开的静态 HTML 页面。正文、图表说明与代码注释均已译为中文，代码结构保持可运行。

## 打开方式

直接双击 `index.html`，或在浏览器中打开：

```bash
xdg-open index.html
```

页面不依赖外部网络。图片、样式、脚本与 KaTeX 字体均在本地目录中，整个 `annotated-diffusion/` 文件夹可以整体移动。

## 目录结构

```text
annotated-diffusion/
├── index.html                 # 完整文章页面
├── README.md
└── assets/
    ├── style.css              # 页面与响应式样式
    ├── article.js             # 目录高亮、复制代码、图片灯箱、阅读进度
    ├── pygments.css           # 代码高亮配色
    ├── favicon.svg
    ├── images/                # 文章配图（WebP/PNG/JPEG/GIF，均为本地文件）
    │   ├── hero.webp
    │   ├── ddpm_paper.webp
    │   ├── diffusion_figure.png
    │   ├── training.png
    │   ├── unet_architecture.webp
    │   ├── output_cats.jpeg
    │   ├── output_cats_verify.png
    │   ├── output_cats_noisy.png
    │   ├── output_cats_noisy_multiple.webp
    │   ├── sampling.png
    │   ├── output.png
    │   └── diffusion-sweater.gif
    └── vendor/
        └── katex/             # 本地 KaTeX 样式与字体（公式在构建时预渲染）
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

本页面为学习交流用途的中文翻译。原文、插图及相关材料的版权归 Hugging Face 与原作者所有。KaTeX 依据 MIT License 使用，许可证见 `assets/vendor/katex/LICENSE`。
