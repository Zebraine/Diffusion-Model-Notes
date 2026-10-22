### 生成问题

假设有一批来自未知分布 $p_{data}(x_0)$ 的图像 $x_0$，我们想要构造一个模型 $p_{\theta}(x_0)$ 来学习这个分布。  
模型应该有能够采样出类似图像的能力。

DDPM 因此引入一组与图像同维度的潜变量 $x_1, x_2, \ldots, x_T$。  
定义两条链：

- 前向过程 $q(x_{1:T}|x_0)$：固定不变，逐步像图像添加高斯噪声。
- 逆向过程 $p_{\theta}(x_{0:T})$：可学习的，逐步去噪声。

前向过程把复杂的数据分布逐渐抹平成一个简单的高斯分布，而逆向过程则试图从这个高斯分布中恢复出原始数据分布。  
生成模型就是完成这个逆向过程的模型。

$$
p_{\theta}(x_0) := p(x_T) \prod_{t=1}^{T} p_{\theta}(x_{t-1}|x_t),\quad p(x_T) = \mathcal{N}(0, I)
$$

训练过程是让 $p_{\theta}(x_0)$ 尽可能接近 $p_{data}(x_0)$。  
但是直接最大化似然函数 $\log p_{\theta}(x_0)$ 是不可行的，因为 $p_{\theta}(x_0)$ 是一个积分：

$$
p_{\theta}(x_0) = \int p_{\theta}(x_{0:T}) dx_{1:T}
$$

所以需要用变分推断，优化一个可计算的上界。

### 前向过程

定义噪声强度 $\beta_1, \ldots, \beta_T$，其中 $\beta_t \in (0, 1)$。  
并记：

$$
\alpha_t := 1 - \beta_t, \quad \bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s
$$

前向过程的一步是：

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

也就是：

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

由于每一步都是线性高斯，所以整条链可以跳步采样：

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

等价的有：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

所以任意噪声级别的 $x_t$ 都可以直接从 $x_0$ 采样，而不需要经过中间的 $x_1, \ldots, x_{t-1}$。  
前向过程没有可学习的参数，因为后验，ELBO和训练目标都依赖于前向过程是已知的线性高斯链。

### 逆向过程与后验

逆向过程的定义是：

$$
p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t))
$$

在 DDPM 中，协方差通常固定为 $\sigma_t^2 I$，网络只学习均值 $\mu_{\theta}$。

后验的目的是在给定 $x_t$ 的情况下，推断出 $x_{t-1}$ 的分布。  
所以需要的单步后验是：

$$
q(x_{t-1}|x_t, x_0)
$$

也就是在已知干净图像 $x_0$ 的情况下，给定噪声图像 $x_t$，推断出前一步的图像 $x_{t-1}$。

#### 为什么可解

由贝叶斯公式可得：

$$
q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}, x_0) q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$

其中分母 $q(x_t|x_0)$ 是已知常数，所以实际正比于分子。

而分子的亮相都是高斯分布：

- $q(x_t|x_{t-1}, x_0) = q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I)$
- $q(x_{t-1}|x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1 - \bar{\alpha}_{t-1}) I)$

两个高斯分布的乘积还是高斯分布，所以后验 $q(x_{t-1}|x_t, x_0)$ 也是高斯分布。  
经过配方后可得：

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

其中

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

所以后验均值可以理解为两个方向的加权融合：
- 从 $x_t$ 倒推：$x_t = \sqrt{\alpha_t} x_{t-1} + 噪声$，所以 $x_{t-1} \approx \frac{1}{\sqrt{\alpha_t}} x_t$，方差为 $\frac{\beta_t}{\alpha_t} I$。
- 从 $x_0$ 正推：$x_t = \sqrt{\bar{\alpha}_t} x_0 + 噪声$，方差为 $(1 - \bar{\alpha}_t) I$。

后验的精度是两个方向的精度之和：
$$
A = \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_t} = \frac{1 - \bar{\alpha}_{t}}{\beta_t (1 - \bar{\alpha}_{t-1})}
$$

所以方差为：
$$
\tilde{\beta}_t = \frac{1}{A} = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}
$$

#### 采样时没有 $x_0$

后验公式需要 $x_0$，但是生成时是没有原图的，因此需要通过网络从 $x_t$ 预测出 $x_0$，然后再代入后验公式。  
由前向过程的公式可得：
$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}
$$  

如果将 $\epsilon$ 更换为网络预测的噪声 $\epsilon_{\theta}(x_t, t)$，就可以得到网络预测的逆向均值：
$$
\mu_{\theta}(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}(x_t, t) \right)
$$

也就是训练时用真实 $x_0$ 定义后验，采样时用网络估计的 $x_0$ 代替。

### 变分上界

直接最大化 $\log p_{\theta}(x_0)$ 是不可行的，因为 $p_{\theta}(x_0)$ 是一个积分：
$$
\log p_{\theta}(x_0) = \log \int p_{\theta}(x_{0:T}) dx_{1:T}
$$  
而变分推断的目标是最大化一个可计算的下界（ELBO）：
$$
\log p_{\theta}(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)} \right] =: \mathcal{L}_{\text{ELBO}}(\theta)
$$

把联合分布和近似后验都按马尔科夫链展开，再利用贝叶斯公式，可以将 ELBO 拆成 3 个部分：
1. 终止项：$D_{KL}(q(x_T|x_0) || p(x_T))$，衡量前向过程的终止分布与高斯分布的差异。
2. 中间项：$\sum_{t=2}^{T} D_{KL}(q(x_{t-1}|x_t, x_0) || p_{\theta}(x_{t-1}|x_t))$，衡量后验与网络预测的逆向分布的差异。
3. 初始项：$-\log p_{\theta}(x_0|x_1)$，衡量网络预测的 $x_0$ 与真实 $x_0$ 的差异。

### 从 KL 到噪声回归

在 $p_{\theta}$ 的方差 $\Sigma_{\theta}(x_t, t)$ 固定为 $\sigma_t^2 I$ 的情况下，KL 散度可以化简为均值的平方误差：
$$
L_{t-1} := D_{KL}(q(x_{t-1}|x_t, x_0) || p_{\theta}(x_{t-1}|x_t)) = \frac{1}{2 \sigma_t^2} \| \tilde{\mu}_t(x_t, x_0) - \mu_{\theta}(x_t, t) \|^2
$$

将 $\tilde{\mu}_t(x_t, x_0)$ 和 $\mu_{\theta}(x_t, t)$ 的公式代入，经过一系列推导，可以得到：
$$
\tilde{\mu}_t(x_t, x_0) - \mu_{\theta}(x_t, t) = \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \left( \epsilon - \epsilon_{\theta}(x_t, t) \right)
$$
所以
$$
L_{t-1} = \frac{\beta_t^2}{2 \sigma_t^2 (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_{\theta}(x_t, t) \|^2
$$

也就是均值之差最终可以转化为噪声之差，训练目标就变成了噪声回归。

DDPM 实际采用的训练目标是：
$$
L_{\text{simple}}(\theta) := \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_{\theta}(x_t, t) \|^2 \right]
$$

这个目标是对每个时间步的 KL 散度加权求和的一个简化版本，忽略了 $\beta_t$ 和 $\sigma_t$ 的系数。  

### $\epsilon$-预测就是 score matching

对每个 t，加噪后的数据分布记作：
$$
q_t(x_t) := \int q(x_t|x_0) p_{data}(x_0) dx_0
$$

定义 score 为对数密度的梯度：
$$
s_t(x_t) := \nabla_{x_t} \log q_t(x_t)
$$

对单个 $x_0$，条件 score 是：
$$
s_t(x_t|x_0) := \nabla_{x_t} \log q(x_t|x_0) = -\frac{1}{1 - \bar{\alpha}_t} (x_t - \sqrt{\bar{\alpha}_t} x_0)
$$

边缘 score 是条件 score 的期望：
$$
s_t(x_t) = \mathbb{E}_{q(x_0|x_t)}[s_t(x_t|x_0)] = -\frac{1}{1 - \bar{\alpha}_t} (x_t - \sqrt{\bar{\alpha}_t} \mathbb{E}_{q(x_0|x_t)}[x_0])
$$

而 $L_{\text{simple}}$ 的均方误差最优解是：
$$
\epsilon_{\theta}^*(x_t, t) = \mathbb{E}_{q(x_0|x_t)}[\epsilon] = \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (x_t - \sqrt{\bar{\alpha}_t} \mathbb{E}_{q(x_0|x_t)}[x_0])
$$

这和 score 的关系是：
$$
s_t(x_t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}^*(x_t, t)
$$

也就是，训练 $\epsilon$-预测网络等价于训练 score 网络。

### 采样 ———— 沿 score 走回数据

在已知 $\mu_{\theta}$ 的情况下，逆向过程的一步是：
$$
x_{t-1} = \mu_{\theta}(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

其中 $\sigma_t^2$ 可以取为 $\tilde{\beta}_t$，也可以取为 0。

不断迭代这个过程，就可以从高斯噪声采样出数据分布的样本。  
同时每一步都保留随机性，保证了采样的多样性。


