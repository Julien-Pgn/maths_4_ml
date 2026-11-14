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
 
We compute the expectation of the estimator found in 1.4.3, using the definition of $\hat{\sigma}^2_{\text{ML}}$:
 
$$
\mathbb{E}\left[\hat{\sigma}^2_{\text{ML}}\right] = \mathbb{E}\left[\frac{1}{N}\sum_{n=1}^{N}\left(x_n - \hat{\mu}_{\text{ML}}\right)^2\right]
$$
 
We expand the square, $(a - b)^2 = a^2 - 2ab + b^2$, and take the constant $\frac{1}{N}$ out of the expectation:
 
$$
\mathbb{E}\left[\hat{\sigma}^2_{\text{ML}}\right] = \frac{1}{N}\,\mathbb{E}\left[\sum_{n=1}^{N}\left(x_n^2 - 2x_n\hat{\mu}_{\text{ML}} + \hat{\mu}_{\text{ML}}^2\right)\right]
$$
 
$\hat{\mu}_{\text{ML}}$ does not depend on $n$, and $\sum_{n=1}^{N} x_n = N\hat{\mu}_{\text{ML}}$ (estimator from 1.4.2), so $\sum_{n} 2x_n\hat{\mu}_{\text{ML}} = 2N\hat{\mu}_{\text{ML}}^2$ and $\sum_{n}\hat{\mu}_{\text{ML}}^2 = N\hat{\mu}_{\text{ML}}^2$:
 
$$
\begin{aligned}
\mathbb{E}\left[\hat{\sigma}^2_{\text{ML}}\right]
&= \frac{1}{N}\left(\mathbb{E}\left[\sum_{n=1}^{N} x_n^2 - 2N\hat{\mu}_{\text{ML}}^2 + N\hat{\mu}_{\text{ML}}^2\right]\right) \\
&= \frac{1}{N}\left(\mathbb{E}\left[\sum_{n=1}^{N} x_n^2 - N\hat{\mu}_{\text{ML}}^2\right]\right)
\end{aligned}
$$
 
By linearity of expectation, and using $\mathbb{E}[X^2] = \operatorname{Var}(X) + \mathbb{E}[X]^2$ for $x_n$ and for $\hat{\mu}_{\text{ML}}$:
 
$$
\mathbb{E}\left[\hat{\sigma}^2_{\text{ML}}\right] = \frac{1}{N}\left(\sum_{n=1}^{N}\left(\operatorname{Var}(x_n) + \left(\mathbb{E}[x_n]\right)^2\right) - N\left(\operatorname{Var}(\hat{\mu}_{\text{ML}}) + \left(\mathbb{E}[\hat{\mu}_{\text{ML}}]\right)^2\right)\right)
$$
 
The samples are i.i.d. with mean $\mu$ and variance $\sigma^2$, so $\operatorname{Var}(x_n) = \sigma^2$ and $\mathbb{E}[x_n] = \mu$. For the sample mean, $\mathbb{E}[\hat{\mu}_{\text{ML}}] = \mu$ and, by independence, $\operatorname{Var}(\hat{\mu}_{\text{ML}}) = \frac{1}{N^2}\sum_{n}\operatorname{Var}(x_n) = \frac{\sigma^2}{N}$:
 
$$
\begin{aligned}
\mathbb{E}\left[\hat{\sigma}^2_{\text{ML}}\right]
&= \frac{1}{N}\left(\sum_{n=1}^{N}\left(\sigma^2 + \mu^2\right) - N\left(\frac{\sigma^2}{N} + \mu^2\right)\right) \\
&= \frac{1}{N}\left(N\sigma^2 + N\mu^2 - \sigma^2 - N\mu^2\right) \\
&= \frac{1}{N}\left(N\sigma^2 - \sigma^2\right)
 = \frac{1}{N}\left(\sigma^2(N - 1)\right)
 = \frac{\sigma^2(N - 1)}{N}
\end{aligned}
$$
 
$$
\boxed{\mathbb{E}\left[\hat{\sigma}^2_{\text{ML}}\right] = \frac{\sigma^2(N - 1)}{N}}
$$
 
Thus $\mathbb{E}\left[\hat{\sigma}^2_{\text{ML}}\right] \neq \sigma^2$, and it is smaller since $\frac{N - 1}{N} < 1$, which demonstrates that $\hat{\sigma}^2_{\text{ML}}$ is biased.

## 2 Linear regression
### 2.1 parameters of linear regression
#### 2.1.1
 
Loss to minimize:
 
$$
\ell(w_1, w_2) = \frac{1}{N}\sum(y_n - w_1 x_{n1} - w_2 x_{n2})^2
$$
 
Partial derivatives (chain rule):
 
$$
\frac{\partial \ell}{\partial w_1} = \frac{1}{N}\sum 2(y_n - w_1 x_{n1} - w_2 x_{n2}) \cdot (-x_{n1})
$$
 
$$
\frac{\partial \ell}{\partial w_2} = \frac{1}{N}\sum 2(y_n - w_1 x_{n1} - w_2 x_{n2}) \cdot (-x_{n2})
$$
 
Setting the derivative with respect to $w_1$ to $0$:
 
$$
\frac{\partial \ell}{\partial w_1} = 0 \;\Rightarrow\; \sum (y_n - w_1 x_{n1} - w_2 x_{n2})(-x_{n1}) = 0
$$
 
$$
\Rightarrow \sum(-x_{n1} y_n + w_1 x_{n1}^2 + w_2 x_{n2} x_{n1}) = 0
$$
 
$$
= \sum(-x_{n1} y_n) + w_1 \sum x_{n1}^2 + w_2 \sum(x_{n2} \cdot x_{n1}) = 0
$$
 
Setting the derivative with respect to $w_2$ to $0$:
 
$$
\frac{\partial \ell}{\partial w_2} = 0 \;\Rightarrow\; \sum(y_n - w_1 x_{n1} - w_2 x_{n2}) \cdot (-x_{n2}) = 0
$$
 
$$
\Rightarrow \sum(-x_{n1} y_n) + w_2 \sum x_{n2}^2 + w_1 \sum(x_{n1} \cdot x_{n2})
$$
 
To keep the notation clean we will define:
 
$$
A = \sum x_{n1}^2 \qquad B = \sum x_{n1} x_{n2} \qquad C = \sum x_{n2}^2 \qquad D = \sum x_{n1} y_n \qquad E = \sum x_{n2} y_n
$$
 
First equation:
 
$$
\Rightarrow -D + w_1 A + w_2 B = 0
$$
 
$$
\Rightarrow w_1 = \frac{D - w_2 B}{A}
$$
 
Second equation:
 
$$
\frac{\partial \ell}{\partial w_2} = -E + w_1 B + w_2 C = 0
$$
 
Substituting $w_1$:
 
$$
\Rightarrow -E + \left(\frac{D - w_2 B}{A}\right) B + w_2 C = 0
$$
 
$$
\Rightarrow -E + \frac{BD - w_2 B^2}{A} + w_2 C = 0
$$
 
$$
\Rightarrow -AE + BD - w_2 B^2 + w_2 AC = 0
$$
 
$$
w_2 (AC - B^2) = AE - BD
$$
 
$$
\boxed{w_2 = \frac{AE - BD}{AC - B^2}}
$$
 
Let's verify: by substituting $w_2$ into $w_1$
 
$$
w_1 = \frac{D - w_2 B}{A} = \frac{D - \left(\frac{AE - BD}{AC - B^2}\right) B}{A}
$$
 
$$
= \frac{\;\frac{D(AC - B^2) - BAE + B^2 D}{AC - B^2}\;}{A} = \frac{DAC - \cancel{DB^2} - BAE + \cancel{B^2 D}}{A(AC - B^2)}
$$
 
$$
= \frac{DAC - BAE}{A(AC - B^2)} = \frac{A(DC - BE)}{A(AC - B^2)}
$$
 
$$
w_1 = \frac{DC - BE}{AC - B^2}
$$
 
#### 2.1.2
 
Let's verify if results from 2.1.1 are identical to (3):
 
For $\mathbf{X} \in \mathbb{R}^{N \times 2}$ and $\mathbf{y} \in \mathbb{R}^{N}$: compute $\mathbf{X}^\top \mathbf{X}$.
 
$$
\mathbf{X} = \begin{pmatrix} x_{1,1} & x_{1,2} \\ \vdots & \vdots \\ x_{N,1} & x_{N,2} \end{pmatrix}
$$
 
We have
 
$$
\mathbf{X}^\top \mathbf{X} = \begin{pmatrix} x_{1,1} & \cdots & x_{N,1} \\ x_{1,2} & \cdots & x_{N,2} \end{pmatrix} \begin{pmatrix} x_{1,1} & x_{1,2} \\ \vdots & \vdots \\ x_{N,1} & x_{N,2} \end{pmatrix}
$$
 
$$
= \begin{pmatrix} \sum x_{n,1}^2 & \sum x_{n,1}\,x_{n,2} \\ \sum x_{n,1}\,x_{n,2} & \sum x_{n,2}^2 \end{pmatrix}
$$
 
$$
\mathbf{X}^\top \mathbf{y} = \begin{pmatrix} x_{1,1} & \cdots & x_{N,1} \\ x_{1,2} & \cdots & x_{N,2} \end{pmatrix} \begin{pmatrix} y_1 \\ \vdots \\ y_N \end{pmatrix} = \begin{pmatrix} \sum x_{n,1}\,y_n \\ \sum x_{n,2}\,y_n \end{pmatrix}
$$
 
Using previous terms:
 
$$
A = \sum x_{n,1}^2 \qquad B = \sum x_{n,1}\,x_{n,2} \qquad C = \sum x_{n,2}^2
$$
 
$$
D = \sum x_{n,1}\,y_n \qquad E = \sum x_{n,2}\,y_n
$$
 
Thus we have
 
$$
\mathbf{X}^\top \mathbf{X} = \begin{pmatrix} A & B \\ B & C \end{pmatrix} \quad\text{and}\quad \mathbf{X}^\top \mathbf{y} = \begin{pmatrix} D \\ E \end{pmatrix}
$$
 
So
 
$$
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$
 
$$
(\mathbf{X}^\top \mathbf{X})^{-1} = \frac{1}{AC - B^2}\begin{pmatrix} C & -B \\ -B & A \end{pmatrix}
$$
 
according to
 
$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$
 
Thus
 
$$
\mathbf{w} = \frac{1}{AC - B^2}\begin{pmatrix} C & -B \\ -B & A \end{pmatrix}\begin{pmatrix} D \\ E \end{pmatrix}
$$
 
$$
\mathbf{w} = \frac{1}{AC - B^2}\begin{pmatrix} CD - BE \\ AE - BD \end{pmatrix} = \begin{pmatrix} \dfrac{CD - BE}{AC - B^2} \\[2ex] \dfrac{AE - BD}{AC - B^2} \end{pmatrix}
$$
 
which corresponds to $\mathbf{w} = \begin{pmatrix} w_1 \\ w_2 \end{pmatrix}$ from previous findings.
 
#### 2.2.1
 
Assuming the $N$ observations $\mathcal{D} = \{(x_1, y_1) \dots (x_N, y_N)\}$ are independent and identically distributed, the joint probability density factorizes as:
 
$$
p(\mathcal{D} \mid w) = \prod p(x_n, y_n \mid w)
$$
 
Applying the product rule mentioned:
 
$$
p(a, b) = p(a \mid b)\,p(b)
$$
 
and noting that $p(x_n)$ doesn't depend on $w$, we can consider it a constant $k$.
 
$$
p(\mathcal{D} \mid w) = \prod p(y_n \mid x_n, w)\,p(x_n)
$$
 
Because $y_n \mid x_n, w \sim \mathcal{N}(w \cdot x_N, \sigma^2)$ the conditional density is
 
$$
p(y_n \mid x_n, w) = \frac{1}{\sqrt{2\pi\sigma^2}}\,e^{\left(-\frac{1(y_n - w \cdot x_n)}{2\sigma^2}\right)}
$$
 
$$
= \frac{1}{2}\ln(2\pi\sigma^2) - \frac{1(y_n - w \cdot x_n)}{2\sigma^2} + \ln(p(x_n)) \quad (\text{constante})
$$
 
$$
= \sum\left(-\frac{1}{2}\ln(2\pi\sigma^2) - \frac{1(y_n - w \cdot x_n)^2}{2\sigma^2} + \text{constante}\right)
$$
 
$$
= -\frac{1}{2}N\ln 2\pi - \frac{1}{2}N\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum(y_n - w \cdot x_n)^2 + \text{cst}
$$
 
#### 2.2.2
 
To find the maximum likelihood estimate $\tilde{w}_{\text{ML}}$ we maximize $l(w)$ with respect to $w$:
With what we find in 2.2.1:
 
$$
\tilde{w}_{\text{ML}} = \arg\max_{w}\left[-\frac{N}{2}\ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum(y_n - w \cdot x_n)^2 + C\right]
$$
 
Remove all terms that don't depend on $w$:
 
$$
\tilde{w}_{\text{ML}} = \arg\max_{w}\left[-\frac{1}{2\sigma^2}\sum_{n=1}^{N}(y_n - w \cdot x_n)^2\right]
$$
 
Since maximizing a negative quantity is equivalent to minimizing its positive counterpart:
 
$$
\tilde{w}_{\text{ML}} = \arg\min_{w}\left[\frac{1}{2\sigma^2}\sum_{n=1}^{N}(y_n - w \cdot x_n)^2\right]
$$
 
Since $\sigma^2 > 0$, the $1/2\sigma^2 > 0$, it just scales the objective without shifting its minimizer, we can remove it from the equation.
 
$$
\tilde{w}_{\text{ML}} = \arg\min_{w}\left[\sum_{n=1}^{N}(y_n - w \cdot x_n)^2\right]
$$
 
almost is equivalent to (2) without $1/N$.