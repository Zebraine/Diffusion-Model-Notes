# DDPM · 模块化 DDPM（CPU）

参考 *The Annotated Diffusion Model* 与 DDPM 原始论文实现的去噪扩散概率模型。代码按职责拆分为多个模块，默认在 CPU 上运行，包含训练、EMA、检查点、采样和一个小型烟雾测试。

> 目录名使用 `DDPM`；Python 包名与 CLI 命令沿用 `dppm`，以保持旧命令兼容。

## 目录结构

```text
src/dppm/
├── config.py          # 数据 / 模型 / 扩散 / 训练 / 采样配置（dataclass）
├── data.py            # 合成数据、ImageFolder、CIFAR-10、HuggingFace datasets
├── diffusion.py       # beta 调度、q_sample、p_losses、DDPM / DDIM 采样
├── ema.py             # 模型权重的指数滑动平均
├── model/
│   ├── blocks.py      # Residual / Downsample / Upsample / ResnetBlock / 时间嵌入
│   ├── attention.py   # 自注意力、线性注意力、PreNorm
│   └── unet.py        # DDPM U-Net
├── train.py           # 训练循环、日志、EMA、检查点、采样预览
├── sample.py          # 从检查点加载并生成图片
├── visualize.py       # 保存图片网格与单张样本
├── utils.py           # 随机种子、设备、日志、JSON 工具
├── cli.py             # dppm train / sample / inspect / smoke
├── __main__.py        # python -m dppm
└── main.py            # 兼容旧入口的薄壳
```

对应关系：

| 代码 | 论文 / 博客 |
|---|---|
| `GaussianDiffusion.q_sample` | 式 (4)，闭式加噪 |
| `GaussianDiffusion.p_losses` | 式 (14)，`L_simple` |
| `GaussianDiffusion.p_sample` | 算法 2，DDPM 逆向一步 |
| `GaussianDiffusion.sample` | 算法 2 的完整循环 |
| `GaussianDiffusion.ddim_sample` | DDIM 快速采样（CPU 推理默认） |
| `UNet` | 博客中的 `Unet`，PixelCNN++ 风格骨干 |
| `EMA` | 博客中的 `EMA` 类 |

## 安装

项目已经配置 PyTorch CPU 源：

```bash
cd code/DDPM
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

如果没有 uv，也可以手动安装 CPU 版 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

如果 `uv` 的默认缓存目录不可写，可以指定仓库内的缓存目录：

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --offline
```

验证：

```bash
dppm inspect --image-size 32 --dim 16 --dim-mults 1,2
```

## 快速开始：2 秒 CPU 烟雾测试

```bash
dppm smoke --output-dir runs/smoke
```

它会：

1. 用程序生成的彩色形状数据训练 2 步（32×32、16 通道、CPU）。
2. 保存检查点与采样预览。
3. 从检查点重新加载 EMA 权重，用 2 步 DDIM 生成 2 张图片。

产物：

```text
runs/smoke/
├── checkpoints/last.pt
├── samples/step_000001.png
├── samples/step_000002.png
├── smoke_sample.png
├── config.json
├── metrics.jsonl
└── summary.json
```

## 训练

### 1. 合成形状（无需下载）

```bash
dppm train \
  --source synthetic \
  --image-size 32 \
  --dim 16 \
  --dim-mults 1,2 \
  --batch-size 4 \
  --max-steps 200 \
  --sample-every 100 \
  --sample-steps 20 \
  --num-sample-images 4 \
  --output-dir runs/synthetic
```

### 2. 本地图片（ImageFolder）

目录下每个子文件夹是一类，例如：

```text
flowers/
├── daisy/*.jpg
├── rose/*.jpg
└── tulip/*.jpg
```

```bash
dppm train \
  --source imagefolder \
  --data-dir /path/to/flowers \
  --image-size 64 \
  --dim 32 \
  --dim-mults 1,2,4 \
  --batch-size 8 \
  --epochs 20 \
  --sample-every 500 \
  --sample-steps 50 \
  --output-dir runs/flowers
```

### 3. CIFAR-10

```bash
dppm train \
  --source cifar10 \
  --cifar-root data \
  --download \
  --image-size 32 \
  --dim 32 \
  --dim-mults 1,2,4 \
  --batch-size 16 \
  --epochs 30 \
  --sample-every 1000 \
  --output-dir runs/cifar10
```

### 4. HuggingFace datasets

```bash
dppm train \
  --source hf \
  --hf-dataset huggan/flowers-102-categories \
  --hf-split train \
  --hf-image-key image \
  --image-size 64 \
  --dim 32 \
  --dim-mults 1,2,4 \
  --batch-size 8 \
  --epochs 20 \
  --output-dir runs/flowers102
```

`--hf-image-key` 需要与数据集的图像列名一致（常见为 `image` 或 `img`）。

## 采样

```bash
dppm sample \
  --checkpoint runs/flowers/checkpoints/last.pt \
  --sampler ddim \
  --steps 50 \
  --num-samples 16 \
  --batch-size 8 \
  --grid-cols 4 \
  --output samples/flowers.png
```

- `--sampler ddim`：默认，50 步即可出图，适合 CPU。
- `--sampler ddpm`：完整 1000 步 DDPM 采样，更慢但对应论文算法 2。
- `--no-ema`：使用原始模型权重，而不是 EMA 权重。
- `--individual`：额外保存每张图。
- `--eta`：DDIM 的随机性参数，默认 `0.0`（确定性采样）。

## 训练输出

```text
runs/<name>/
├── checkpoints/last.pt   # 模型、EMA、优化器、配置、epoch/step
├── samples/step_XXXXXX.png
├── config.json           # 完整训练配置
├── metrics.jsonl         # step / epoch / loss / lr / elapsed
└── summary.json
```

断点续训：

```bash
dppm train ... --resume runs/flowers/checkpoints/last.pt
```

续训会从检查点保存时的下一个 epoch 开始，并沿用命令行中的当前配置；模型与扩散配置必须与检查点一致。检查点通过 `torch.load(..., weights_only=False)` 读取，请只加载自己信任的文件。

## 设计说明

- **CPU 优先**：默认 `--device cpu`，默认模型 `dim=32`、`dim_mults=(1,2,4)`，64×64 输入约 260 万参数。
- **固定前向方差**：线性 `beta` 调度（也支持 cosine / quadratic / sigmoid）。
- **ε-预测 + 简化目标**：默认 Huber 损失，`L_simple` 对应论文式 (14)。
- **EMA**：默认衰减 `0.9999`；采样时优先使用 EMA 权重。
- **DDIM 快速采样**：CPU 上不必跑满 1000 步；DDPM 采样仍保留。
- **尺寸约束**：`image_size` 必须能被 `2 ** (len(dim_mults) - 1)` 整除。例如 64×64 配 `1,2,4`，32×32 配 `1,2`。
- **无网络副作用**：`data.py` 只在显式选择 `hf` 时才导入 `datasets`；不会像旧草案那样在 import 时下载示例图片。

## 常用命令

```bash
dppm --help
dppm train --help
dppm sample --help
dppm inspect --image-size 64 --dim 32 --dim-mults 1,2,4
dppm smoke
```

## 开发检查

```bash
python -m compileall -q src
ruff check src
```

## 版权与来源

- 论文：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)（Ho、Jain、Abbeel，NeurIPS 2020）
- 博客：[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)（Niels Rogge、Kashif Rasul）
- 本实现参考博客与论文重写整理，仅用于学习交流。
