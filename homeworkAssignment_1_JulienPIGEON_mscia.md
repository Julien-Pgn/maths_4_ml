# Homework assignment 1:
 
Author: Julien Pigeon for his ML class at CentraleSupelec (MScIA)
 
## 1 Fundamental concepts
 
### 1.1 Bayes theorem
 
Based on the text we know that:
$$
P(+ \mid M) = 0.95 \qquad P(+ \mid \bar{M}) = 0.01 \qquad P(M) = 0.001 \qquad P(\bar{M}) = 1 - P(M) = 0.999
$$
 
And with the Bayes theorem we can get:
$$
P(M \mid +) = \frac{P(+ \mid M)\,P(M)}{P(+ \mid M)\,P(M) + P(+ \mid \bar{M})\,P(\bar{M})}
$$
 
 
$$
= \frac{0.95 \times 0.001}{0.95 \times 0.001 + 0.01 \times 0.999} = 0.087
$$
 
Thus a person with a positive test has only about an 8.7% risk of being infected.
 
 
### 1.2 Matrix calculus
 
Let's start by writing $f$ as a double sum over the components of $\mathbf{x}$:
 
$$
f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x} = \sum_{i=1}^{D} \sum_{j=1}^{D} A_{ij}\, x_i\, x_j
$$
 
We differentiate with respect to one component $x_k$, using the product rule:
 
$$
\frac{\partial (x_i x_j)}{\partial x_k} = \frac{\partial x_i}{\partial x_k}\, x_j + x_i\, \frac{\partial x_j}{\partial x_k} = \delta_{ik}\, x_j + x_i\, \delta_{jk}
$$
 
 where: 
$\delta_{ik} = 1$ if $i = k$, and $0$ otherwise. Since the $A_{ij}$ are constants:
 
$$
\frac{\partial f(\mathbf{x})}{\partial x_k} = \sum_{i=1}^{D} \sum_{j=1}^{D} A_{ij} \left( \delta_{ik}\, x_j + x_i\, \delta_{jk} \right)
$$
 
$$
= \sum_{i=1}^{D} \sum_{j=1}^{D} A_{ij}\, \delta_{ik}\, x_j + \sum_{i=1}^{D} \sum_{j=1}^{D} A_{ij}\, x_i\, \delta_{jk}
$$
 
$\delta_{ik}$ is 0 unless $i = k$ so in the first double sum, only the terms with $i = k$ remain. Same for $\delta_{jk}$ in the second one with $j = k$.
 
$$
\Rightarrow \frac{\partial f(\mathbf{x})}{\partial x_k} = \sum_{j=1}^{D} A_{kj}\, x_j + \sum_{i=1}^{D} A_{ik}\, x_i
$$
 
Using $(\mathbf{A}^\top)_{ki} = A_{ik}$:
 
$$
\frac{\partial f(\mathbf{x})}{\partial x_k} = \sum_{j=1}^{D} A_{kj}\, x_j + \sum_{i=1}^{D} (\mathbf{A}^\top)_{ki}\, x_i = \sum_{j=1}^{D} \left( A_{kj} + (\mathbf{A}^\top)_{kj} \right) x_j
$$
 
which is the $k^{\text{th}}$ entry of the matrix-vector product $(\mathbf{A} + \mathbf{A}^\top)\,\mathbf{x}$. Thus, stacking all $D$ partial derivatives into the gradient vector gives:
 
$$
\frac{\partial\, \mathbf{x}^\top \mathbf{A} \mathbf{x}}{\partial \mathbf{x}} = (\mathbf{A} + \mathbf{A}^\top)\, \mathbf{x}
$$
 
### 1.3 Constrained optimization
 
The constraint is $g(x_1, x_2) = x_1 \cdot x_2 = 4$. Since the product equals $4$, we have $x_2 \neq 0$, so we can divide by $x_2$:
 
$$
x_1 \cdot x_2 = 4 \iff x_1 = \frac{4}{x_2}
$$
 
We substitute $x_1$ in $f$, which gives a function $h$ of the single variable $x_2$:
 
$$
h(x_2) = f\left(\frac{4}{x_2}, x_2\right) = \left(\frac{4}{x_2}\right)^2 + x_2^2 = \frac{16}{x_2^2} + x_2^2
$$
 
We differentiate with respect to $x_2$ 
 
$$
h'(x_2) = 2x_2 - 16 \cdot \frac{2x_2}{x_2^4} = 2x_2 - \frac{32}{x_2^3}
$$
 
Setting the derivative to $0$ (and multiplying both sides by $\frac{x_2^3}{2}$):
 
$$
\begin{aligned}
h'(x_2) = 2x_2 - \frac{32}{x_2^3} = 0
&\iff 2x_2 = \frac{32}{x_2^3} \\
&\iff x_2 \cdot x_2^3 = \frac{32}{2} \\
&\iff x_2^4 = 16 = 2^4
\end{aligned}
$$
 
So $x_2 = 2$ or $x_2 = -2$.
 
Second derivative, to check that these points are minima:
 
$$
h''(x_2) = \frac{d}{dx_2}\left(2x_2 - \frac{32}{x_2^3}\right) = 2 + 32 \cdot \frac{3x_2^2}{x_2^6} = 2 + \frac{96}{x_2^4} > 0
$$
 
because anything to the power of 4 would be positive. 
 
For $x_2 = 2$:
 
$$
x_1 = \frac{4}{2} = 2 = x_2, \qquad f(2, 2) = 2^2 + 2^2 = 8
$$
 
For $x_2 = -2$:
 
$$
x_1 = \frac{4}{-2} = -2, \qquad f(-2, -2) = (-2)^2 + (-2)^2 = 8
$$
 
The minimum of $f$ subject to $g(x_1, x_2) = 4$ is $8$, reached at $(x_1, x_2) = (2, 2)$ and $(x_1, x_2) = (-2, -2)$.
 
### 1.4 Maximum likelihood estimation
 
 
#### 1.4.1 Log-likelihood
 
Because we assume that we have N independent and identically distributed (i.i.d.) samples $\mathbf{x} = \{x_1, x_2, \dots, x_N\}$, the likelihood is the product of the individual densities:
 
$$
p(\mathbf{x} \mid \mu, \sigma^2) = \prod_{n=1}^{N} \mathcal{N}(x_n \mid \mu, \sigma^2)
$$
 
Using the ln of this product turns it in a sum:
 
$$
\begin{aligned}
\ln p(\mathbf{x} \mid \mu, \sigma^2)
&= \ln \prod_{n=1}^{N} \mathcal{N}(x_n \mid \mu, \sigma^2)
 = \sum_{n=1}^{N} \ln \mathcal{N}(x_n \mid \mu, \sigma^2) \\
&= \sum_{n=1}^{N} \ln\left( \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{1}{2}\frac{(x_n-\mu)^2}{\sigma^2} \right) \right) \\
&= \sum_{n=1}^{N} \left( \ln\left( (2\pi\sigma^2)^{-1/2} \right) + \ln\left( \exp\left( -\frac{(x_n-\mu)^2}{2\sigma^2} \right) \right) \right)
\end{aligned}
$$
 
 
$$
\begin{aligned}
\ln p(\mathbf{x} \mid \mu, \sigma^2)
&= \sum_{n=1}^{N} \left( -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x_n-\mu)^2}{2\sigma^2} \right) \\
&= -\frac{N}{2}\ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2
\end{aligned}
$$
 
$$
\boxed{\mathcal{L}(\mu, \sigma^2) = -\frac{N}{2}\ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2}
$$
 
#### 1.4.2 Maximum likelihood estimator of $\mu$
 
Variable: $\mu$. Constants: $\sigma^2$, $N$, $x_1, \dots, x_N$. The first two terms of $\mathcal{L}$ do not depend on $\mu$, so their derivative is $0$.
 
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
\sum_{n=1}^{N} x_n = N\mu \iff \mu = \frac{1}{N}\sum_{n=1}^{N} x_n
$$
 
$$
\boxed{\hat{\mu}_{\text{ML}} = \frac{1}{N}\sum_{n=1}^{N} x_n}
$$
 
The maximum likelihood estimator of $\mu$ is the sample mean.
 
 
#### 1.4.3 Maximum likelihood estimator of $\sigma^2$
 
Let's derive with respect to $\sigma^2$
 
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
 
#### 1.4.a Bonus
 
We compute the expectation of the estimator found in 1.4.3, using the definition of $\hat{\sigma}^2_{\text{ML}}$:
 
$$
\mathbb{E}\left[\hat{\sigma}^2_{\text{ML}}\right] = \mathbb{E}\left[\frac{1}{N}\sum_{n=1}^{N}\left(x_n - \hat{\mu}_{\text{ML}}\right)^2\right]
$$
  
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
 
### 2.1 Parameters of linear regression
 
#### 2.1.1
 
All sums run over $n = 1, \dots, N$.
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
\Rightarrow \sum(-x_{n1} y_n) + w_1 \sum x_{n1}^2 + w_2 \sum(x_{n2} \cdot x_{n1}) = 0
$$
 
Setting the derivative with respect to $w_2$ to $0$:
 
$$
\frac{\partial \ell}{\partial w_2} = 0 \;\Rightarrow\; \sum(y_n - w_1 x_{n1} - w_2 x_{n2}) \cdot (-x_{n2}) = 0
$$
 
$$
\Rightarrow \sum(-x_{n2} y_n) + w_2 \sum x_{n2}^2 + w_1 \sum(x_{n1} \cdot x_{n2}) = 0
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
\Rightarrow -E + w_1 B + w_2 C = 0
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
 
Let's verify by substituting $w_2$ into $w_1$
 
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
 
Valid if $AC - B^2 \neq 0$, since it is the denominator of both $w_1$ and $w_2$. On the way we also divided by $A$, so we assumed $A \neq 0$ too.
 
#### 2.1.2
 
Let's verify if result from 2.1.1 is identical to (3):
 
With $\mathbf{X} \in \mathbb{R}^{N \times 2}$ and $\mathbf{y} \in \mathbb{R}^{N}$: compute $\mathbf{X}^\top \mathbf{X}$.
 
$$
\mathbf{X} = \begin{pmatrix} x_{11} & x_{12} \\ \vdots & \vdots \\ x_{N1} & x_{N2} \end{pmatrix}
$$
 
We have
 
$$
\mathbf{X}^\top \mathbf{X} = \begin{pmatrix} x_{11} & \cdots & x_{N1} \\ x_{12} & \cdots & x_{N2} \end{pmatrix} \begin{pmatrix} x_{11} & x_{12} \\ \vdots & \vdots \\ x_{N1} & x_{N2} \end{pmatrix}
$$
 
$$
= \begin{pmatrix} \sum x_{n1}^2 & \sum x_{n1}\,x_{n2} \\ \sum x_{n1}\,x_{n2} & \sum x_{n2}^2 \end{pmatrix}
$$
 
$$
\mathbf{X}^\top \mathbf{y} = \begin{pmatrix} x_{11} & \cdots & x_{N1} \\ x_{12} & \cdots & x_{N2} \end{pmatrix} \begin{pmatrix} y_1 \\ \vdots \\ y_N \end{pmatrix} = \begin{pmatrix} \sum x_{n1}\,y_n \\ \sum x_{n2}\,y_n \end{pmatrix}
$$
 
Using previous terms:
 
$$
A = \sum x_{n1}^2 \qquad B = \sum x_{n1}\,x_{n2} \qquad C = \sum x_{n2}^2
$$
 
$$
D = \sum x_{n1}\,y_n \qquad E = \sum x_{n2}\,y_n
$$
 
Thus we have
 
$$
\mathbf{X}^\top \mathbf{X} = \begin{pmatrix} A & B \\ B & C \end{pmatrix} \quad\text{and}\quad \mathbf{X}^\top \mathbf{y} = \begin{pmatrix} D \\ E \end{pmatrix}
$$
 
So:
 
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
 
 This formula only works if $ad - bc \neq 0$ (the determinant), otherwise we divide by $0$ and the matrix has no inverse. Here $ad - bc = AC - B^2$, so we need $AC - B^2 \neq 0$, the same condition as in 2.1.1.
 
Thus
 
$$
\mathbf{w} = \frac{1}{AC - B^2}\begin{pmatrix} C & -B \\ -B & A \end{pmatrix}\begin{pmatrix} D \\ E \end{pmatrix}
$$
 
$$
\mathbf{w} = \frac{1}{AC - B^2}\begin{pmatrix} CD - BE \\ AE - BD \end{pmatrix} = \begin{pmatrix} \dfrac{CD - BE}{AC - B^2} \\[2ex] \dfrac{AE - BD}{AC - B^2} \end{pmatrix}
$$
 
which corresponds to $\mathbf{w} = \begin{pmatrix} w_1 \\ w_2 \end{pmatrix}$ from previous findings.
 
 
### 2.2 Least square loss
 
#### 2.2.1
 
Assuming the $N$ observations $\mathcal{D} = \{(x_1, y_1) \dots (x_N, y_N)\}$ are independent and identically distributed, the joint probability density factorizes as:
 
$$
p(\mathcal{D} \mid w) = \prod p(x_n, y_n \mid w)
$$
 
Applying the product rule mentioned in the hint:
 
$$
p(a, b) = p(a \mid b)\,p(b)
$$
 
and noting that $p(x_n)$ doesn't depend on $w$:
 
$$
p(\mathcal{D} \mid w) = \prod p(y_n \mid x_n, w)\,p(x_n)
$$
 
Because $y_n \mid x_n, w \sim \mathcal{N}(w \cdot x_n, \sigma^2)$ the conditional density is
 
$$
p(y_n \mid x_n, w) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_n - w \cdot x_n)^2}{2\sigma^2}\right)
$$
 
Taking the log of one term of the product, $p(y_n \mid x_n, w)\,p(x_n)$:
 
$$
\ln\big(p(y_n \mid x_n, w)\,p(x_n)\big) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(y_n - w \cdot x_n)^2}{2\sigma^2} + \ln(p(x_n))
$$
 
$\ln(p(x_n))$ does not depend on $w$, so it is a constant, which we call $c_n = \ln(p(x_n))$.
 
The log turns the product into a sum, so the log-likelihood $\mathcal{L}(w) = \ln p(\mathcal{D} \mid w)$ is:
 
$$
\mathcal{L}(w) = \sum\left(-\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(y_n - w \cdot x_n)^2}{2\sigma^2} + c_n\right)
$$
 
$$
= -\frac{N}{2}\ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum(y_n - w \cdot x_n)^2 + \text{cst}
$$
 
where $\text{cst} = \sum c_n = \sum \ln(p(x_n))$ is the sum of all these constants, so it does not depend on $w$ either.
 
 
#### 2.2.2
 
To find the maximum likelihood estimate $\hat{w}_{\text{ML}}$ we maximize $\mathcal{L}(w)$ with respect to $w$:
With what we found in 2.2.1:
 
$$
\hat{w}_{\text{ML}} = \arg\max_{w}\left[-\frac{N}{2}\ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum(y_n - w \cdot x_n)^2 + \text{cst}\right]
$$
 
Remove all terms that don't depend on $w$:
 
$$
\hat{w}_{\text{ML}} = \arg\max_{w}\left[-\frac{1}{2\sigma^2}\sum(y_n - w \cdot x_n)^2\right]
$$
 
Maximizing $-g(w)$ is the same as minimizing $g(w)$:
 
$$
\hat{w}_{\text{ML}} = \arg\min_{w}\left[\frac{1}{2\sigma^2}\sum(y_n - w \cdot x_n)^2\right]
$$
 
$\frac{1}{2\sigma^2}$ is a positive constant, so it does not change the argmin and we can drop it:
 
$$
\hat{w}_{\text{ML}} = \arg\min_{w}\left[\sum(y_n - w \cdot x_n)^2\right]
$$
 
which is (2) without $1/N$.
 
Since $\frac{1}{N} > 0$, multiplying by $\frac{1}{N}$ does not change the argmin, so maximizing the log-likelihood is exactly equivalent to minimizing (2).
 
## 3 Logistic regression
 
### 3.1
 
We write $p = P(y = 1 \mid \mathbf{x})$, so $1 - p = P(y = 0 \mid \mathbf{x})$. The log-odds is linear in $\mathbf{x}$:
 
$$
\mathbf{w}^\top \mathbf{x} = \ln\left(\frac{p}{1-p}\right)
$$
 
Taking the exponential of both sides and solving for $p$:
 
$$
e^{\mathbf{w}^\top \mathbf{x}} = \frac{p}{1-p} \quad\Rightarrow\quad p = (1-p)\,e^{\mathbf{w}^\top \mathbf{x}} = e^{\mathbf{w}^\top \mathbf{x}} - p\,e^{\mathbf{w}^\top \mathbf{x}}
$$
 
$$
p + p\,e^{\mathbf{w}^\top \mathbf{x}} = e^{\mathbf{w}^\top \mathbf{x}}
$$
 
$$
p\left(1 + e^{\mathbf{w}^\top \mathbf{x}}\right) = e^{\mathbf{w}^\top \mathbf{x}}
$$
 
$$
p = \frac{e^{\mathbf{w}^\top \mathbf{x}}}{1 + e^{\mathbf{w}^\top \mathbf{x}}}
$$
 
Dividing the numerator and the denominator by $e^{\mathbf{w}^\top \mathbf{x}}$:
 
$$
p = \frac{e^{\mathbf{w}^\top \mathbf{x}} / e^{\mathbf{w}^\top \mathbf{x}}}{\left(1 + e^{\mathbf{w}^\top \mathbf{x}}\right) / e^{\mathbf{w}^\top \mathbf{x}}} = \frac{1}{e^{-\mathbf{w}^\top \mathbf{x}} + 1}
$$
 
So:
 
$$
P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}} = \sigma(\mathbf{w}^\top \mathbf{x}) \quad \text{with} \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$
 
where $\sigma$ is the standard logistic sigmoid function.
 
### 3.2
 
From 3.1:
 
$$
p = \sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}}
$$
 
and so:
 
$$
1 - p = 1 - \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}} = \frac{e^{-\mathbf{w}^\top \mathbf{x}}}{1 + e^{-\mathbf{w}^\top \mathbf{x}}}
$$
 
For sample $n$, we write $\hat{y}_n$ for the predicted probability:
 
$$
\hat{y}_n = P(y_n = 1 \mid \mathbf{x}_n, \mathbf{w}) = \sigma(\mathbf{w}^\top \mathbf{x}_n)
$$
 
Since $y_n$ is $0$ or $1$, both cases fit in one formula:
 
$$
P(y_n \mid \mathbf{x}_n, \mathbf{w}) = \hat{y}_n^{\,y_n} \left(1 - \hat{y}_n\right)^{1 - y_n}
$$
 
Assuming the observations are i.i.d.:
 
$$
L(\mathbf{w}) = \prod_{n=1}^{N} P(y_n \mid \mathbf{x}_n, \mathbf{w}) = \prod_{n=1}^{N} \hat{y}_n^{\,y_n} \left(1 - \hat{y}_n\right)^{1 - y_n}
$$
 
Taking the log:
 
$$
\ln\big(L(\mathbf{w})\big) = \sum_{n=1}^{N} \Big[\, y_n \ln \hat{y}_n + (1 - y_n) \ln(1 - \hat{y}_n) \Big]
$$
 
As in 2.2.2, maximizing $\ln(L(\mathbf{w}))$ is the same as minimizing its opposite, and multiplying by $\frac{1}{N}$ does not change the argmin:
 
$$
\hat{\mathbf{w}}_{\text{ML}} = \arg\max_{\mathbf{w}} \ln\big(L(\mathbf{w})\big) = \arg\min_{\mathbf{w}} \Big(-\ln\big(L(\mathbf{w})\big)\Big)
$$
 
$$
= \arg\min_{\mathbf{w}} \left( -\frac{1}{N} \sum_{n=1}^{N} \Big[\, y_n \ln \sigma(\mathbf{w}^\top \mathbf{x}_n) + (1 - y_n) \ln\big(1 - \sigma(\mathbf{w}^\top \mathbf{x}_n)\big) \Big] \right)
$$
 
which is the logistic loss (binary cross-entropy).
 
 
## 4 Clustering
 
### 4.1 K-Means
 
With the assignments $a_{n,k}$ fixed, only the terms of cluster $k$ depend on $\mathbf{c}_k$, so for a fixed $k$ we can write:
 
$$
J(\mathbf{c}_k) = \frac{1}{N}\sum_{n=1}^{N} a_{n,k}\,\|\mathbf{x}_n - \mathbf{c}_k\|_2^2
$$
 
Derivative with respect to $\mathbf{c}_k$:
 
$$
\frac{\partial J}{\partial \mathbf{c}_k} = \frac{1}{N}\sum_{n=1}^{N} 2\left(a_{n,k}(\mathbf{x}_n - \mathbf{c}_k)\right)\cdot(-1)
$$
 
$$
= -\frac{2}{N}\sum_{n=1}^{N} a_{n,k}(\mathbf{x}_n - \mathbf{c}_k)
$$
 
At the minimum, the derivative is zero:
 
$$
-\frac{2}{N}\sum_{n=1}^{N} a_{n,k}(\mathbf{x}_n - \mathbf{c}_k) = 0
$$
 
$$
\Rightarrow \sum_{n=1}^{N} a_{n,k}(\mathbf{x}_n - \mathbf{c}_k) = 0
$$
 
$$
\Rightarrow \sum_{n=1}^{N}\left(a_{n,k}\,\mathbf{x}_n - a_{n,k}\,\mathbf{c}_k\right) = 0
$$
  
$$
\Rightarrow \sum_{n=1}^{N} a_{n,k}\,\mathbf{x}_n - \mathbf{c}_k\sum_{n=1}^{N} a_{n,k} = 0
$$
 
Assuming cluster $k$ is not empty, i.e. $\sum_{n=1}^{N} a_{n,k} \neq 0$, we can divide:
 
$$
\Rightarrow \mathbf{c}_k = \frac{\sum_{n=1}^{N} a_{n,k}\,\mathbf{x}_n}{\sum_{n=1}^{N} a_{n,k}}
$$
 
So $\mathbf{c}_k$ is the mean of the points assigned to cluster $k$.
 
### 4.2 Hierarchical clustering and Levenshtein distance
 
#### 4.2.1
 
We check the four properties of a distance, for all strings $m$, $n$ and $p$:
 
1) $d(m,n)$ is the number of character insertions, deletions or substitutions necessary to transform $m$ into $n$. Thus the distance is defined as a count of discrete edit operations, which cannot be negative. Therefore:
$$
d(m,n) \geq 0
$$
 
2) If $m = n$, then no insertion, deletion or substitution is needed, so $d(m,n) = 0$. Conversely, $d(m,n) = 0$ means that no operation is needed to transform $m$ into $n$, so $n$ is $m$ unchanged. Thus:
$$
d(m,n) = 0 \iff m = n
$$
 
3) Every edit operation has an exact inverse operation (e.g. inserting an "a" is undone by deleting this "a", deleting an "a" is undone by inserting it back, substituting "a" by "b" is undone by substituting "b" by "a"). Therefore, any sequence of $k = d(m,n)$ operations transforming $m$ into $n$ can be reversed step by step to transform $n$ into $m$ with $k$ operations, so $d(n,m) \leq d(m,n)$. Swapping the roles of $m$ and $n$ gives $d(m,n) \leq d(n,m)$. Thus:
$$
d(m,n) = d(n,m)
$$
 
4) Let's transform $m$ into $n$ with an intermediate step going through $p$. $d(m,p)$ is the minimum number of operations to transform $m$ into $p$, and $d(p,n)$ is the minimum number of operations to transform $p$ into $n$. Concatenating these two sequences of operations produces a sequence of $d(m,p) + d(p,n)$ operations transforming $m$ into $n$. Since $d(m,n)$ is by definition the minimum number of operations to transform $m$ into $n$, it is less than or equal to the number of operations of this sequence. Thus:
$$
d(m,n) \leq d(m,p) + d(p,n)
$$
 
The four axioms hold, so the Levenshtein distance is a distance.
 
#### 4.2.2
 
With single linkage, $D(C_i, C_j) = \min_{x \in C_i,\, y \in C_j} d(x, y)$. 
 
We take $d(x,y) = |x - y|$ and 3 non-overlapping clusters:
 
$$
C_1 = \{0\} \qquad C_2 = \{1, 2, 3\} \qquad C_3 = \{4, 5\}
$$
 
A distance would satisfy $D(C_1, C_3) \le D(C_1, C_2) + D(C_2, C_3)$. Here:
 
$$
D(C_1, C_3) = \min_{x \in \{0\},\, y \in \{4, 5\}} |x - y| = |0 - 4| = 4
$$
 
$$
D(C_1, C_2) = \min_{x \in \{0\},\, y \in \{1, 2, 3\}} |x - y| = |0 - 1| = 1
$$
 
$$
D(C_2, C_3) = \min_{x \in \{1, 2, 3\},\, y \in \{4, 5\}} |x - y| = |3 - 4| = 1
$$
 
Thus
 
$$
D(C_1, C_3) > D(C_1, C_2) + D(C_2, C_3) \quad \text{because} \quad 4 > 1 + 1
$$
 
Thus the triangle inequality axiom is not satisfied which demonstrates that the single-linkage criterion is not a distance.