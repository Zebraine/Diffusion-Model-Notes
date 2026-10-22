## Diffusion Model Notes

Personal notes on diffusion models.

- 论文库（中文精译 / 精读）：打开 [`papers/index.html`](papers/index.html)。
- 新增论文、构建方式与目录约定：见 [`papers/README.md`](papers/README.md)。
- 构建工具：`cd tools && npm install && node build.mjs`。

当前收录：

1. **图解扩散模型**（The Annotated Diffusion Model）— Hugging Face Blog 中文全文翻译。
2. **去噪扩散概率模型**（Denoising Diffusion Probabilistic Models, DDPM）— NeurIPS 2020 中文精译。
3. **DDPM 的数学原理** — 从变分下界到 score matching 的原理长文。
4. **DDPM 的工程实现** — U-Net、EMA、DDIM、检查点与优化的工程笔记。

## 代码实现

- CPU 版 DDPM（模块化训练 / 采样 / 检查点）：[`code/DDPM`](code/DDPM)，用法见 [`code/DDPM/README.md`](code/DDPM/README.md)。
