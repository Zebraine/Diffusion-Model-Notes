## 去噪扩散概率模型 DDPM

**扩散概率模型** 是一个用变分推断训练的参数化马尔科夫链，在有限步之后产生与数据分布匹配的样本。  
扩散模型的采样过程是一种 **渐进式解码**，类似自回归解码。

### 背景

扩散模型是形如：

$$
p_\theta(x_{0}) := \int p_\theta(x_{0:T})dx_{1:T}
$$

的潜变量模型。

联合分布 $p\theta(x_{0:T})$ 称为 **逆向过程**，可以被定义为一条从 $p(x_T) = \mathcal{N}(0, I)$ 出发，带学习高斯转移的马尔科夫链：

$$
p_\theta(x_{0:T}) := p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t),\qquad p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

扩散模型与其它潜变量模型的区别是，近似后验 $q(x_{1:T}|x_0)$ 被称为 **前向过程** 或 **扩散过程**，被固定为一条马尔科夫链，按照方差调度 $\beta_1, \ldots, \beta_T$ 逐步向数据分布添加噪声：

$$
q(x_{1:T}|x_0) := \prod_{t=1}^{T}q(x_t|x_{t-1}),\qquad q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

训练通过优化负对数似然的常用变分上界完成：

$$
\mathcal{L}_{\text{VLB}} := \mathbb{E} \left[-\log p_\theta(x_0)\right] \leq \mathbb{E}_q\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = \mathbb{E}_q\left[-\log p_\theta(x_T) - \sum_{t>1}\log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} - \log p_\theta(x_0|x_1)\right] := L
$$

前向过程的方差 $\beta_t$ 可以通过重参数化学习，也可以作为超参数固定。  
前向过程有一个重要性质，允许在任意时间步 $t$ 直接从数据样本 $x_0$ 中采样 $x_t$：

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I),\qquad \bar{\alpha}_t := \prod_{s=1}^{t}\alpha_s,\quad \alpha_t := 1 - \beta_t
$$

因此可以通过随机梯度下降优化 $L$ 的随机项。通过把 $L$ 重写成 KL 散度之和，可以进一步降低方差：

$$
L = \underbrace{D_{KL}(q(x_T|x_0)||p_\theta(x_T))}_{L_T} + \sum_{t>1}\underbrace{D_{KL}(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}
$$

其中 $L_{t-1}$ 部分用 KL 散度直接比较 $q(x_{t-1}|x_t, x_0)$ 和 $p_\theta(x_{t-1}|x_t)$，也就是逆向过程与前向过程的后验分布。  
在给定 $x_0$ 的情况下，前向过程的后验分布是一个高斯分布：

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\right)
$$

其中

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}\beta_t}{1-\alpha_t}x_0 + \frac{\sqrt{1-\beta_t}(1-\alpha_t)}{1-\alpha_t}x_t,\qquad \tilde{\beta}_t = \frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
$$

### 扩散模型与去噪自编码器

#### 前向过程

关于前向过程的学习，DDPM 将前向方差 $\beta_t$ 作为超参数固定，因此近似后验 $q$ 没有可学习的参数。

#### 逆向过程

首先令 $\Sigma_\theta(x_t, t) = \sigma_t^2 I$，其中 $\sigma_t^2$ 是一个固定的超参数。  
实验中 $\sigma_t^2$ 可以被设置为 $\beta_t$ 或 $\tilde{\beta}_t$，得到的结果相似。

为了表示均值 $\mu_\theta(x_t, t)$，首先研究 $L_t$ 的表示，基于前面的研究可以得出：

$$
L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2\right] + C
$$

进一步地，利用重参数化技巧，可以将 $\tilde{\mu}_t(x_t, x_0)$ 表示为：

$$
\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right),\qquad \epsilon \sim \mathcal{N}(0, I)
$$

代入公式可得：

$$
L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\left\|\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right) - \mu_\theta(x_t, t)\right\|^2\right] + C
$$

而如果我们令 $\mu_\theta(x_t, t)$ 预测噪声 $\epsilon$，即：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)
$$

则可以得到：

$$
L_{t-1} = \mathbb{E}_q\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha_t)}\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right] + C
$$

忽略与训练无关的常数项，并令 $\sigma_t^2 = \beta_t$，则可以得到一个简单的训练目标：

$$
L_{t-1} = \mathbb{E}_q\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

所以训练实际上是在学习一个去噪自编码器 $\epsilon_\theta(x_t, t)$，它从噪声样本 $x_t$ 中预测出添加的噪声 $\epsilon$。  
也就是模型通过预测添加的噪声，来学习如何恢复所需的样本 $x_0$。

### 结论

DDPM 通过分析前向过程和逆向过程的关系，通过分析马尔科夫链的性质，提出了一个有效的训练目标，使得模型能够学习如何从噪声中恢复出原始数据。  
并通过不断化简，将单步反向预测均值的学习问题转化为预测噪声的学习问题，从而简化了训练过程。

但是，DDPM 也存在明显问题，由于每次预测只针对单步的噪声进行优化，导致使用 DDPM 生成真实图像时需要大量的采样步骤，通常需要上百步甚至上千步才能生成高质量的图像，这使得生成过程非常缓慢。  
同时大量的采样步骤也会累积误差，导致生成的图像质量下降。
