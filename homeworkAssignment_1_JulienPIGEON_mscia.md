# Homework assignment 1:

Author: Julien Pigeon for his ML class at CentraleSupelec (MScIA)

## 1. Fundamental concepts

### 1.4 Maximum likelyhood estimation


#### 1.4.1 Log-likelihood

Because we assume that we have N independent and identically distributed (i.i.d.) samples x = {x1, x2, . . . , xN }, the likelihood is the product of the individual densities:

$$
p(\mathbf{x} \mid \mu, \sigma^2) = \prod_{n=1}^{N} \mathcal{N}(x_n \mid \mu, \sigma^2)
$$

Taking the natural logarithm turns the product into a sum, using $\ln(ab) = \ln a + \ln b$:

$$
\begin{aligned}
\ln p(\mathbf{x} \mid \mu, \sigma^2)
&= \ln \prod_{n=1}^{N} \mathcal{N}(x_n \mid \mu, \sigma^2)
 = \sum_{n=1}^{N} \ln \mathcal{N}(x_n \mid \mu, \sigma^2) \\
&= \sum_{n=1}^{N} \ln\left( \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{1}{2}\frac{(x_n-\mu)^2}{\sigma^2} \right) \right) \\
&= \sum_{n=1}^{N} \left( \ln\left( (2\pi\sigma^2)^{-1/2} \right) + \ln\left( \exp\left( -\frac{(x_n-\mu)^2}{2\sigma^2} \right) \right) \right)
\end{aligned}
$$

We use $\ln(a^k) = k \ln a$ and $\ln(\exp(u)) = u$:

$$
\begin{aligned}
\ln p(\mathbf{x} \mid \mu, \sigma^2)
&= \sum_{n=1}^{N} \left( -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x_n-\mu)^2}{2\sigma^2} \right) \\
&= -\frac{N}{2}\ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2
\end{aligned}
$$

The first two terms do not depend on $n$, so summing them $N$ times multiplies them by $N$. We write $\mathcal{L}(\mu, \sigma^2) = \ln p(\mathbf{x} \mid \mu, \sigma^2)$:

$$
\boxed{\mathcal{L}(\mu, \sigma^2) = -\frac{N}{2}\ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2}
$$

#### 1.4.2 Maximum likelihood estimator of $\mu$

Variable: $\mu$. Constants: $\sigma^2$, $N$, $x_1, \dots, x_N$. The first two terms of $\mathcal{L}$ do not depend on $\mu$, so their derivative is $0$. By the chain rule, $\frac{\partial}{\partial \mu}(x_n-\mu)^2 = 2(x_n-\mu)\cdot(-1)$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mu}
&= -\frac{1}{2\sigma^2}\sum_{n=1}^{N} \frac{\partial}{\partial \mu}(x_n-\mu)^2
 = -\frac{1}{2\sigma^2}\sum_{n=1}^{N} 2(x_n-\mu)\cdot(-1) \\
&= \frac{1}{\sigma^2}\sum_{n=1}^{N}(x_n-\mu)
\end{aligned}
$$

Setting the derivative to $0$ (and multiplying both sides by $\sigma^2 > 0$):

$$
\begin{aligned}
\frac{1}{\sigma^2}\sum_{n=1}^{N}(x_n-\mu) = 0
&\iff \sum_{n=1}^{N} x_n - \sum_{n=1}^{N}\mu = 0 \\
&\iff \sum_{n=1}^{N} x_n - N\mu = 0
\end{aligned}
$$

$$
\boxed{\hat{\mu}_{\text{ML}} = \frac{1}{N}\sum_{n=1}^{N} x_n}
$$

The maximum likelihood estimator of $\mu$ is the sample mean.

Check with a small example: $\mathbf{x} = (1, 5, 3)$, $N = 3$:

$$
\hat{\mu}_{\text{ML}} = \frac{1+5+3}{3} = \frac{9}{3} = 3
$$

and indeed $\sum_{n}(x_n - 3) = -2 + 2 + 0 = 0$.

#### 1.4.3 Maximum likelihood estimator of $\sigma^2$

Variable: $\sigma^2$, treated as one block (we write $s = \sigma^2$ in our head). Constants: $\mu$, $N$, $x_1, \dots, x_N$. The term $-\frac{N}{2}\ln(2\pi)$ is constant, and we use $\frac{d}{ds}\ln s = \frac{1}{s}$ and $\frac{d}{ds}\frac{1}{s} = -\frac{1}{s^2}$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \sigma^2}
&= -\frac{N}{2}\cdot\frac{1}{\sigma^2} - \frac{1}{2}\cdot\left(-\frac{1}{(\sigma^2)^2}\right)\sum_{n=1}^{N}(x_n-\mu)^2 \\
&= -\frac{N}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{n=1}^{N}(x_n-\mu)^2
\end{aligned}
$$

Setting the derivative to $0$ and multiplying both sides by $2(\sigma^2)^2 > 0$:

$$
\begin{aligned}
-\frac{N}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{n=1}^{N}(x_n-\mu)^2 = 0
&\iff -N\sigma^2 + \sum_{n=1}^{N}(x_n-\mu)^2 = 0 \\
&\iff \sigma^2 = \frac{1}{N}\sum_{n=1}^{N}(x_n-\mu)^2
\end{aligned}
$$

Both derivatives must be $0$ at the same time, so we replace $\mu$ by the estimator $\hat{\mu}_{\text{ML}}$ found in 1.4.2:

$$
\boxed{\hat{\sigma}^2_{\text{ML}} = \frac{1}{N}\sum_{n=1}^{N}\left(x_n-\hat{\mu}_{\text{ML}}\right)^2}
$$

Check with the same example: $\mathbf{x} = (1, 5, 3)$, $\hat{\mu}_{\text{ML}} = 3$:

$$
\hat{\sigma}^2_{\text{ML}} = \frac{(1-3)^2 + (5-3)^2 + (3-3)^2}{3} = \frac{4+4+0}{3} = \frac{8}{3}
$$

#### 1.4.a: bonus

Calculating E is equivalent to asking: on average on all sample do i get sigma**2 ? 



