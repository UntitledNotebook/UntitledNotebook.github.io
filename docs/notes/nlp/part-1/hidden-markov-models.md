# Hidden Markov models

We will mainly follow Chapter 5, “Inference for Markov Models,” from *Random Processes for Engineers*, adding examples from the lecture slides.

## Basics of estimation theory

**What is estimation?** Suppose a parameter $\theta$ is to be estimated from a random variable $Y$. An *estimator* of $\theta$ is a function $\hat{\theta}$ that gives the *estimate* $\hat{\theta}(y)$ for each observed value $y$.

### Two views: ML and Bayes

**Definition 1 (maximum likelihood).** Suppose the distribution of $Y$ is known for each admissible parameter $\theta$. Fix the observed value $y$. Its likelihood as a function of $\theta$ and the corresponding maximum likelihood (ML) estimate are

$$
L(\theta;y)=
\begin{cases}
p_Y(y\mid\theta), & Y \text{ discrete},\\
f_Y(y\mid\theta), & Y \text{ continuous},
\end{cases}
\qquad
\hat{\theta}_{\mathrm{ML}}(y)\in\arg\max_{\theta} L(\theta;y).
\tag{1}
$$

ML can impose hard constraints by restricting the admissible values of $\theta$. A Bayesian model can also express prior beliefs by giving the unknown quantity a probability distribution.

**Definition 2 (Bayes estimator).** Let $Z$ be the quantity to estimate and $Y$ the observation, with a known joint distribution. Let $C(z,\hat z)$ be the cost of reporting $\hat z$ when the true value is $z$. A Bayes estimator $\hat Z=g^*(Y)$ minimizes the average cost over estimation rules $g$:

$$
g^*\in\arg\min_g \mathbb E \left[C\left(Z,g(Y)\right)\right].
\tag{2}
$$

The **prior** is the distribution of $Z$ before observing $Y$. The **posterior** is the conditional distribution of $Z$ given $Y=y$. For discrete $Z$, their probability mass functions are $p_Z(z)$ and $p_{Z\mid Y}(z\mid y)$. For jointly continuous $Z,Y$, their densities are $f_Z(z)$ and $f_{Z\mid Y}(z\mid y)$.

### MMSE and MAP estimators

Under squared-error loss, the minimum mean squared error (MMSE) estimate is the posterior mean:

$$
C(z,\hat z)=(z-\hat z)^2,\qquad
\hat Z_{\mathrm{MMSE}}(y)=\mathbb E[Z\mid Y=y].
\tag{3}
$$

For discrete $Z$, zero-one loss $C(z,\hat z)=\mathbf 1_{\{z\ne\hat z\}}$ assigns unit cost to an incorrect estimate. Given $Y=y$, reporting $z$ has expected cost $1-p_{Z\mid Y}(z\mid y)$. Minimizing this cost gives the *maximum a posteriori* (MAP) estimate:

$$
\hat Z_{\mathrm{MAP}}(y)\in\arg\max_z p_{Z\mid Y}(z\mid y).
\tag{4}
$$

Returning to the parameter notation, let $\Theta$ and $Y$ be discrete random variables. For an observation with $p_Y(y)>0$, Bayes' rule gives

$$
p_{\Theta\mid Y}(\theta\mid y)
=\frac{p_\Theta(\theta)\,p_Y(y\mid\theta)}{p_Y(y)},
\qquad
p_Y(y)=\sum_{\theta'}p_\Theta(\theta')\,p_Y(y\mid\theta').
\tag{5}
$$

Here $p_Y(y\mid\theta)$ denotes the likelihood under $\Theta=\theta$. With $y$ fixed, the denominator $p_Y(y)$ is constant across candidate values of $\theta$, so (5) gives

$$
\begin{aligned}
\hat\Theta_{\mathrm{MAP}}(y)
&\in\arg\max_\theta p_{\Theta\mid Y}(\theta\mid y)\\
&=\arg\max_\theta \frac{p_\Theta(\theta)\,p_Y(y\mid\theta)}{p_Y(y)}\\
&=\arg\max_\theta p_\Theta(\theta)\,p_Y(y\mid\theta).
\end{aligned}
\tag{6}
$$

Recall that the maximum likelihood estimate satisfies

$$
\hat{\theta}_{\mathrm{ML}}(y)\in\arg\max_{\theta} p_Y(y\mid\theta).
\tag{7}
$$

If the prior is uniform over the admissible parameter values, its factor is constant too. MAP and ML then have the same maximizers.

For jointly continuous $\Theta$ and $Y$, the MAP estimate maximizes the posterior density:

$$
\hat\Theta_{\mathrm{MAP}}(y)
\in\arg\max_\theta f_{\Theta\mid Y}(\theta\mid y)
=\arg\max_\theta f_\Theta(\theta)\,f_Y(y\mid\theta).
\tag{8}
$$

## The expectation-maximization algorithm

The expectation-maximization (EM) algorithm is an iterative method for maximum-likelihood estimation in models with latent variables.

**Notation.**

- $\theta$ is the parameter to estimate. $\theta^{(k)}$ is its value at iteration $k$.
- $X$ is the complete data. Its probability mass function $p_{\mathrm{cd}}(x\mid\theta)$ is specified for each $\theta$.
- $Y=h(X)$ is the observed random vector, and $y$ is its observed value. Write $\mathcal F_y=\{x:h(x)=y\}$ for all complete-data values consistent with $y$.
- When $X=(Y,Z)$, $Z$ denotes the unobserved component. In this case $h(y,z)=y$, so summing over $\mathcal F_y$ amounts to summing over $z$ with $y$ fixed.

The observed-data likelihood is the total probability of the complete-data values that yield $y$:

$$
p(y\mid\theta)=\sum_{x\in\mathcal F_y}p_{\mathrm{cd}}(x\mid\theta).
\tag{9}
$$

In some applications, there can be a very large number of terms in (9), making direct numerical maximization of $p(y\mid \theta)$ with respect to $\theta$ computationally infeasible (i.e., infeasible to compute $\hat{\theta}_{\text{ML}}(y)$).

### Algorithm: Expectation-maximization

An observation $y$ is given, along with an initial estimate $\theta^{(0)}$.

**(Expectation step)** Compute $Q(\theta\mid\theta^{(k)})$ for all $\theta$, where

$$
Q(\theta\mid\theta^{(k)})=\mathbb E\!\left[\log p_{\mathrm{cd}}(X\mid\theta)\mid y,\theta^{(k)}\right]. \tag{10}
$$

**(Maximization step)** Compute $\theta^{(k+1)}\in\arg\max_{\theta} Q(\theta\mid\theta^{(k)})$.

A common source of confusion is the different roles of $\theta$ and $\theta^{(k)}$: the posterior weights are computed using $\theta^{(k)}$, while the complete-data log-likelihood is evaluated at $\theta$.

$$
\begin{aligned}
Q(\theta\mid\theta^{(k)})
=\sum_{x\in\mathcal F_y}
\frac{p_{\mathrm{cd}}(x\mid\theta^{(k)})}{p(y\mid\theta^{(k)})}
\log p_{\mathrm{cd}}(x\mid\theta),
\qquad p(y\mid\theta^{(k)})>0.
\end{aligned}
\tag{11}
$$

### Convergence of the EM algorithm

Suppose that the complete data pmf can be factored as

$$
p_{X, Y}(x, y\mid \theta) = p_{\text{cd}}(x\mid \theta) \mathbf{1}_{\{h(x)=y\}}=p(y\mid \theta)k(x\mid y, \theta)
\tag{12}
$$

Assume the following conditions hold:

1. The map $\theta\mapsto\log p(y\mid\theta)$ is differentiable.
2. For every $\bar\theta$, the conditional expectation $\mathbb E[\log k(X\mid y,\bar\theta)\mid y,\bar\theta]$ is finite.
3. For each fixed $\theta$, the map $\theta'\mapsto D\!\left(k(\cdot\mid y,\theta)\,\|\,k(\cdot\mid y,\theta')\right)$ is differentiable.
4. For each fixed $\theta'$, the map $\theta\mapsto D\!\left(k(\cdot\mid y,\theta)\,\|\,k(\cdot\mid y,\theta')\right)$ is continuous.

If $p(y\mid\theta^{(0)})>0$, the EM iterates satisfy $p(y\mid\theta^{(k+1)})\ge p(y\mid\theta^{(k)})$ for every $k$. Every limit point $\theta^*$ of the parameter sequence is stationary for the likelihood:

$$
\left.\nabla_\theta p(y\mid\theta)\right|_{\theta=\theta^*}=0.
\tag{13}
$$

Here we only prove the nondecreasing part. Using the factorization

$$
p_{\text{cd}}(x\mid \theta)=p(y\mid \theta) k(x \mid y, \theta)
\tag{14}
$$

for $x \in \mathcal{F}_{y}$,

$$
\begin{aligned}
Q(\theta\mid\theta^{(k)})
&=\mathbb E\!\left[\log p_{\mathrm{cd}}(X\mid\theta)\,\middle|\,y,\theta^{(k)}\right]\\
&=\log p(y\mid\theta)
 +\mathbb E\!\left[\log k(X\mid y,\theta)\,\middle|\,y,\theta^{(k)}\right]\\
&=\log p(y\mid\theta)
 +\mathbb E\!\left[\log\frac{k(X\mid y,\theta)}{k(X\mid y,\theta^{(k)})}\,\middle|\,y,\theta^{(k)}\right]+\mathbb E\!\left[\log k(X\mid y,\theta^{(k)})\,\middle|\,y,\theta^{(k)}\right]\\
&=\log p(y\mid\theta)
 -D\!\left(k(\cdot\mid y,\theta^{(k)})\,\|\,k(\cdot\mid y,\theta)\right)+\mathbb E\!\left[\log k(X\mid y,\theta^{(k)})\,\middle|\,y,\theta^{(k)}\right].
\end{aligned}
\tag{15}
$$

Notice that the last term $\mathbb{E}\left[\log k(X \mid y, \theta^{(k)}) \middle| y, \theta^{(k)} \right]$ is constant with respect to $\theta$. The maximization step is equivalent to:

$$
	\theta^{(k+1)} = \arg \max_{\theta} \left[ \log p(y \mid \theta) - D \left( k(\cdot \mid y, \theta^{(k)} ) \| k(\cdot \mid y, \theta) \right)  \right].
\tag{16}
$$

Therefore,

$$
	\log p(y\mid \theta^{(k+1)}) - D\left( k(\cdot \mid y, \theta^{(k)})\| k(\cdot \mid y, \theta^{(k+1)}) \right) \geq \log p(y \mid \theta^{(k)}) - D \left( k(\cdot \mid y, \theta^{(k)}) \| k(\cdot \mid y, \theta^{(k)}) \right) = \log p(y \mid \theta^{(k)}).
\tag{17}
$$

### Example: Gaussian mixture model

An observed random vector $Y\in\mathbb R^d$ follows a mixture of $J$ Gaussian distributions. Let $Z\in\{1,\ldots,J\}$ be the unobserved component label. The parameter is $\theta=(\pi,\mu,\Sigma)$, where

- $\pi=(\pi_1,\ldots,\pi_J)$ contains the **mixing weights**. $\pi_j=P(Z=j)$
- $\mu=(\mu_1,\ldots,\mu_J)$ contains the **component means**. $\mu_j=\mathbb E[Y\mid Z=j]\in\mathbb R^d$.
- $\Sigma=(\Sigma_1,\ldots,\Sigma_J)$ contains the **component covariance matrices**. $\Sigma_j=\operatorname{Cov}(Y\mid Z=j) \in \mathbb{R}^{d \times d}.$

Thus $Y\mid Z=j\sim\mathcal N(\mu_j,\Sigma_j), Z \sim \text{Categorical}(\pi)$. Next, we derive the EM algorithm steps in an attempt to compute the maximum likelihood estimate of $\theta$ given independent observations $y^{(1)}, y^{(2)}, \dots, y^{(n)}$.

Let $X = (Z^{(1)}, Y^{(1)}, Z^{(2)}, Y^{(2)}, \dots, Z^{(n)}, Y^{(n)})$. Then we have

$$
\begin{aligned}
p_{\mathrm{cd}}(X\mid\theta)&=\prod_{i=1}^{n}\pi_{Z^{(i)}}\,\mathcal N\!\left(Y^{(i)}\mid\mu_{Z^{(i)}},\Sigma_{Z^{(i)}}\right) \\
&= \prod_{i=1}^n \prod_{j=1}^J \left( \pi_{j} \mathcal{N} \left( Y^{(i)} \mid \mu_{j}, \Sigma_{j} \right)  \right) ^{\mathbf{1}\{Z^{(i)}=j\}}.
\end{aligned}
\tag{18}
$$

For $i=1,\ldots,n$ and $j=1,\ldots,J$, define:

$$
\begin{aligned}
r_{ij}^{(k)}
&=\mathbb E\!\left[\mathbf 1_{\{Z^{(i)}=j\}}\,\middle|\,Y^{(1)}=y^{(1)},\ldots,Y^{(n)}=y^{(n)},\theta^{(k)}\right]\\
&=P\!\left(Z^{(i)}=j\mid Y^{(i)}=y^{(i)},\theta^{(k)}\right)\\
&=\frac{\pi_j^{(k)}\,\mathcal N\!\left(y^{(i)}\mid\mu_j^{(k)},\Sigma_j^{(k)}\right)}
        {\sum_{\ell=1}^{J}\pi_\ell^{(k)}\,\mathcal N\!\left(y^{(i)}\mid\mu_\ell^{(k)},\Sigma_\ell^{(k)}\right)}.
\end{aligned}
\tag{19}
$$

Let $y=(y^{(1)},\ldots,y^{(n)})$ denote the observed sample. Only the component indicators are random:

$$
\begin{aligned}
Q(\theta\mid\theta^{(k)})
&=\mathbb E\!\left[\log p_{\mathrm{cd}}(X\mid\theta)\,\middle|\,y,\theta^{(k)}\right]\\
&=\sum_{i=1}^{n}\sum_{j=1}^{J}
  \mathbb E\!\left[\mathbf 1_{\{Z^{(i)}=j\}}\,\middle|\,y,\theta^{(k)}\right]
  \log\!\left[\pi_j\,\mathcal N\!\left(y^{(i)}\mid\mu_j,\Sigma_j\right)\right]\\
&=\sum_{i=1}^{n}\sum_{j=1}^{J}r_{ij}^{(k)}
  \left[\log\pi_j+\log\mathcal N\!\left(y^{(i)}\mid\mu_j,\Sigma_j\right)\right].
\end{aligned}
\tag{20}
$$

Keep track of what is being conditioned on and which variables the expectation is taken over. Confusing the two can lead to incorrect updates and make the optimization unnecessarily difficult.

In the M-step,

$$
\begin{aligned}
\arg\max_{\theta} Q(\theta\mid\theta^{(k)})
&=\arg\max_{\pi,\mu,\Sigma}
\sum_{i=1}^{n}\sum_{j=1}^{J} r_{ij}^{(k)}
\left[
\log\pi_j-\frac12\log\det\Sigma_j
-\frac12(y^{(i)}-\mu_j)^\top\Sigma_j^{-1}(y^{(i)}-\mu_j)
\right].
\end{aligned}
\tag{21}
$$

Both maximizations use the GMM constraints $\pi_j\ge 0$, $\sum_{j=1}^{J}\pi_j=1$, and $\Sigma_j\succ 0$ for each component.

- Mixing weights update

    $$
    \boxed{\pi_j^{(k+1)}=\frac{1}{n}\sum_{i=1}^{n}r_{ij}^{(k)}.}
    \tag{22}
    $$

??? note "Matrix Cookbook identities used below"

    For fixed vectors $a,b$ and matrix $B$, the vector and matrix gradients are

    $$
    \begin{aligned}
    \nabla_x(x^\top Bx)&=(B+B^\top)x &&\text{(81)},\\
    \nabla_X\log|\det X|&=X^{-\top} &&\text{(57)},\\
    \nabla_X(a^\top X^{-1}b)&=-X^{-\top}ab^\top X^{-\top} &&\text{(61)}.
    \end{aligned}
    $$

    Here $X^{-\top}=(X^{-1})^\top$. For a positive-definite covariance matrix, $\Sigma_j^{-\top}=\Sigma_j^{-1}$. [Petersen and Pedersen, *The Matrix Cookbook*, 2012, pp. 9–11](https://course.ccs.neu.edu/cs6220f17/ssl/maths/petersen-pederson-matrix-cookbook.pdf#page=9)

- Mean update

    $$
    \nabla_{\mu_j}Q
    =\Sigma_j^{-1}\sum_{i=1}^{n}r_{ij}^{(k)}(y^{(i)}-\mu_j).
    \tag{23}
    $$

    Set this gradient to zero and multiply by $\Sigma_j$.

    $$
    \boxed{\mu_j^{(k+1)}=\frac{\sum_{i=1}^{n}r_{ij}^{(k)}y^{(i)}}{\sum_{i=1}^{n}r_{ij}^{(k)}}.}
    \tag{24}
    $$

- Covariance update

    $$
    \nabla_{\Sigma_j}Q
    =-\frac12\left(\sum_{i=1}^{n}r_{ij}^{(k)}\right)\Sigma_j^{-1}
    +\frac12\Sigma_j^{-1}
    \left(\sum_{i=1}^{n}r_{ij}^{(k)}(y^{(i)}-\mu_j^{(k+1)})(y^{(i)}-\mu_j^{(k+1)})^\top\right)\Sigma_j^{-1}.
    \tag{25}
    $$

    Set the gradient to zero and multiply on both the left and the right by $\Sigma_j$.

    $$
    \boxed{\Sigma_j^{(k+1)}
    =\frac{\displaystyle\sum_{i=1}^{n}r_{ij}^{(k)}
    (y^{(i)}-\mu_j^{(k+1)})(y^{(i)}-\mu_j^{(k+1)})^\top}
    {\displaystyle\sum_{i=1}^{n}r_{ij}^{(k)}}.}
    \tag{26}
    $$

#### NumPy implementation

```python
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def gmm_em(
    Y: FloatArray,                  # (n, d)
    pi_init: FloatArray,            # (J,)
    mu_init: FloatArray,            # (J, d)
    sigma_init: FloatArray,         # (J, d, d)
    max_iter: int = 100,
    tol: float | None = None,
    reg_covar: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit a full-covariance GMM using EM.

    tol=None runs exactly max_iter iterations. Otherwise, stop when
    the absolute change in total observed-data log-likelihood is < tol.
    max_iter must be nonnegative; tol, if supplied, must be positive;
    reg_covar must be nonnegative.

    Returns:
        pi:     (J,) mixing weights.
        mu:     (J, d) means.
        Sigma:  (J, d, d) covariances.
        r:      (n, J) responsibilities under the returned parameters.
        loglik: (t + 1,) log-likelihoods, including initialization,
                where t is the number of completed EM iterations.
    """
    n, d = Y.shape
    J = pi_init.shape[0]
    pi = np.array(pi_init, dtype=np.float64, copy=True)       # (J,)
    mu = np.array(mu_init, dtype=np.float64, copy=True)       # (J, d)
    Sigma = np.array(sigma_init, dtype=np.float64, copy=True) # (J, d, d)
    eye = np.eye(d, dtype=np.float64)                         # (d, d)

    def e_step(
        weights: FloatArray,    # (J,)
        means: FloatArray,      # (J, d)
        covs: FloatArray,       # (J, d, d)
    ) -> tuple[FloatArray, float]:
        log_joint = np.empty((n, J), dtype=np.float64)  # (n, J)
        for j in range(J):
            L = np.linalg.cholesky(covs[j])            # (d, d)
            diff = Y - means[j]                       # (n, d)
            z = np.linalg.solve(L, diff.T)             # (d, n)
            mahal = np.sum(z * z, axis=0)              # (n,)
            log_det = 2.0 * np.log(np.diag(L)).sum()
            log_joint[:, j] = np.log(weights[j]) - 0.5 * (
                d * np.log(2.0 * np.pi) + log_det + mahal
            )

        # Stable log(sum_j pi_j * N(y_i | mu_j, Sigma_j)).
        shift = log_joint.max(axis=1, keepdims=True)   # (n, 1)
        log_density = shift + np.log(
            np.exp(log_joint - shift).sum(axis=1, keepdims=True)
        )                                                     # (n, 1)
        responsibilities = np.exp(log_joint - log_density)     # (n, J)
        return responsibilities, float(log_density.sum())

    r, ll = e_step(pi, mu, Sigma)
    history: list[float] = [ll]

    for _ in range(max_iter):
        # M-step: keep r fixed throughout all three updates.
        counts = r.sum(axis=0)                       # (J,)
        pi = counts / n                             # (J,)
        mu = (r.T @ Y) / counts[:, None]             # (J, d)
        diff = Y[None, :, :] - mu[:, None, :]        # (J, n, d)
        Sigma = np.einsum(
            "jn,jnd,jne->jde", r.T, diff, diff
        ) / counts[:, None, None]                   # (J, d, d)
        Sigma += reg_covar * eye                     # (J, d, d)

        # E-step under the updated parameters.
        r, ll = e_step(pi, mu, Sigma)
        history.append(ll)
        if tol is not None and abs(history[-1] - history[-2]) < tol:
            break

    loglik = np.asarray(history, dtype=np.float64)   # (t + 1,)
    return pi, mu, Sigma, r, loglik
```

#### Two-dimensional example

<figure>
  <img src="../assets/gmm_em.gif" alt="Gaussian mixture EM in two dimensions, from initialization to convergence">
</figure>

## Hidden Markov Models

A hidden Markov model describes a sequence of observations generated by a latent Markov chain. We will see an example of an HMM applied to part-of-speech tagging.

**Notation.** Let $T\ge 1$ be the number of observation times. The complete data are $X=(Y,Z)$.

- $Z=(Z_1,\ldots,Z_T)$ is the hidden, time-homogeneous Markov chain. Each $Z_t$ takes values in the state space $\mathcal{S}$, with $N_s=|S|$.
- $Y=(Y_1,\ldots,Y_T)$ is the observed sequence. Each $Y_t$ takes one of $N_o$ possible values. Given the entire state sequence $Z=z$, the observations are conditionally independent, and the distribution of $Y_t$ depends only on $z_t$.

The HMM parameter is $\boxed{\theta=(\pi,A,B)}$.

| Parameter | Meaning | Entries |
|---|---|---|
| $\pi$ | Initial distribution on $S$ | $\pi_i=P(Z_1=i)$ |
| $A=(a_{ij})\in\mathbb R^{N_s\times N_s}$ | One-step transition matrix | $a_{ij}=P(Z_{t+1}=j\mid Z_t=i)$ |
| $B=(b_{i\ell})\in\mathbb R^{N_s\times N_o}$ | Observation generation matrix | $b_{i\ell}=P(Y_t=\ell\mid Z_t=i)$ |

<figure>
  <img src="../assets/hajek-fig-5-1-hmm.svg" alt="HMM with initial distribution π, transitions A between hidden states, and emissions B from each state to its observation">
</figure>

The pmf of the complete data, for a given choice of $\theta$, is

$$
p_{\mathrm{cd}}(y,z\mid\theta)
=\pi_{z_1}\prod_{t=1}^{T-1}a_{z_t z_{t+1}}\prod_{t=1}^{T}b_{z_t y_t}. \tag{27}
$$

We consider the following three estimation tasks:

1. Given the observed data $y_{1: T}$ and $\theta$, compute the conditional distribution of state $z$.
2. Given the observed data $y_{1 : T}$ and $\theta$, compute the most likely hidden state sequence $z_{1: T}$.
3. Given the observed data $y_{1:T}$, compute the maximum likelihood estimate of $\theta$.

### Posterior state probabilities and the forward-backward algorithm

In this section, we aim to provide an efficient way to compute

$$
	P(Z_{t}=i \mid Y_{1}=y_{1}, \dots, Y_{T}=y_{T}, \theta)
\tag{28}
$$

and

$$
	P(Z_{t}=i,Z_{t+1}=j\mid Y_{1}=y_{1},\dots,Y_{T}=y_{T},\theta).
\tag{29}
$$

For each $i\in\mathcal S$ and $1\le t\le T$, define the **forward variable** $\alpha_i(t)$ and the **backward variable** $\beta_i(t)$ by

$$
\begin{aligned}
\alpha_i(t)&\triangleq P(Y_1=y_1,\ldots,Y_t=y_t,Z_t=i\mid\theta),\\
\beta_i(t)&\triangleq P(Y_{t+1}=y_{t+1},\ldots,Y_T=y_T\mid Z_t=i,\theta).
\end{aligned}
\tag{30}
$$

At the endpoints, $\alpha_i(1)=\pi_i b_{i y_1}$ and $\beta_i(T)=1$. We leave the derivation of the recurrences for $\alpha, \beta$ to the reader. Next, we see how $\alpha, \beta$ help to find the two probabilities we want.

Note that

$$
\begin{aligned}
P(Z_t=i,Y_1=y_1,\ldots,Y_T=y_T\mid\theta)
&=P(Z_t=i,Y_{1:t}=y_{1:t}\mid\theta)\,
  P(Y_{t+1:T}=y_{t+1:T}\mid Z_t=i,Y_{1:t}=y_{1:t},\theta)\\
&=P(Z_t=i,Y_{1:t}=y_{1:t}\mid\theta)\,
  P(Y_{t+1:T}=y_{t+1:T}\mid Z_t=i,\theta)\\
&=\alpha_i(t)\,\beta_i(t).
\end{aligned}
\tag{31}
$$

We can derive the conditional distribution of the state at time $t$, given all the observations:

$$
\begin{aligned}
\gamma_i(t)&\triangleq P(Z_t=i\mid Y_1=y_1,\ldots,Y_T=y_T,\theta)\\
&=\frac{P(Z_t=i,Y_1=y_1,\ldots,Y_T=y_T\mid\theta)}{P(Y_1=y_1,\ldots,Y_T=y_T\mid\theta)}\\
&=\frac{\alpha_i(t)\beta_i(t)}{\sum_{j\in\mathcal S}\alpha_j(t)\beta_j(t)}.
\end{aligned}
\tag{32}
$$

From the above, for any $i$ such that $\gamma_{i}(t) > 0$,

$$
	P(Y_{1}=y_{1},\dots,Y_{T}=y_{T}\mid \theta) = \frac{\alpha_{i}(t)\beta_{i}(t)}{\gamma_{i}(t)}. \tag{33}
$$

Similarly,

$$
\begin{aligned}
&P(Z_t=i,Z_{t+1}=j,Y_1=y_1,\ldots,Y_T=y_T\mid\theta)\\
&=P(Z_t=i,Y_{1:t}=y_{1:t}\mid\theta)\\
&\quad{}\cdot P(Z_{t+1}=j,Y_{t+1}=y_{t+1}\mid Z_t=i,Y_{1:t}=y_{1:t},\theta)\\
&\quad{}\cdot P(Y_{t+2:T}=y_{t+2:T}\mid Z_t=i,Z_{t+1}=j,Y_{1:t+1}=y_{1:t+1},\theta)\\
&=\alpha_i(t)\,a_{ij}\,b_{j y_{t+1}}\,\beta_j(t+1).
\end{aligned}
\tag{34}
$$

From this, we derive the conditional distribution of a pair of states at consecutive times $t$ and $t+1$, given all observations:

$$
\begin{aligned}
\xi_{ij}(t)&\triangleq P(Z_t=i,Z_{t+1}=j\mid Y_1=y_1,\ldots,Y_T=y_T,\theta)\\
&=\frac{P(Z_t=i,Z_{t+1}=j,Y_1=y_1,\ldots,Y_T=y_T\mid\theta)}
        {P(Y_1=y_1,\ldots,Y_T=y_T\mid\theta)}\\
&=\frac{\alpha_i(t)a_{ij}b_{j y_{t+1}}\beta_j(t+1)}
        {\displaystyle\sum_{i',j'\in\mathcal S}
          \alpha_{i'}(t)a_{i'j'}b_{j' y_{t+1}}\beta_{j'}(t+1)}\\
&=\frac{\gamma_i(t)a_{ij}b_{j y_{t+1}}\beta_j(t+1)}{\beta_i(t)}
  \qquad\text{if }\beta_i(t)>0.
\end{aligned}
\tag{35}
$$

where the final expression is derived using (33).

**Algorithm (Forward-backward).** Given the observations $y_{1:T}$ and parameters $\theta=(\pi,A,B)$, make two passes over the sequence.

1. **Forward pass.** Initialize $\alpha_i(1)=\pi_i b_{i y_1}$ for each $i\in\mathcal S$. For $t=1,\ldots,T-1$, compute

    $$
    \alpha_j(t+1)=\sum_{i\in\mathcal S}\alpha_i(t)a_{ij}b_{j y_{t+1}}. \tag{36}
    $$

2. **Backward pass.** Initialize $\beta_i(T)=1$ for each $i\in\mathcal S$. For $t=T,T-1,\ldots,2$, compute

    $$
    \beta_i(t-1)=\sum_{j\in\mathcal S}a_{ij}b_{j y_t}\beta_j(t). \tag{37}
    $$

3. **Compute the posteriors.**

    $$
    \begin{align}
    P(Z_t=i\mid Y_{1:t}=y_{1:t},\theta)
    &=\frac{\alpha_i(t)}{\sum_{j\in\mathcal S}\alpha_j(t)}, \tag{38}\\
    \gamma_i(t)\triangleq P(Z_t=i\mid Y_{1:T}=y_{1:T},\theta)
    &=\frac{\alpha_i(t)\beta_i(t)}{\sum_{j\in\mathcal S}\alpha_j(t)\beta_j(t)}. \tag{39}
    \end{align}
    $$

    $$
    \begin{align}
    \xi_{ij}(t) &\triangleq P(Z_{t}=i,Z_{t+1}=j \mid Y_{1}=y_{1},\dots,Y_{T}=y_{T},\theta) \notag\\
    &= \frac{\alpha_i(t)a_{ij}b_{j y_{t+1}}\beta_j(t+1)}
         {\displaystyle\sum_{u,v\in\mathcal S}\alpha_u(t)a_{uv}b_{v y_{t+1}}\beta_v(t+1)} \tag{40}\\
         &= \frac{\gamma_{i}(t)a_{ij}b_{jy_{t+1}}\beta_{j}(t+1)}{\beta_{i}(t)}. \tag{41}
    \end{align}
    $$

### Most likely state sequence: Viterbi algorithm

Suppose the parameter $\theta = (\pi, A, B)$ is known, and that $Y = (Y_{1},\dots,Y_{T})$ is observed as $y=(y_{1},\dots,y_{T})$. We seek the MAP estimate $\hat{Z}_{\text{MAP}}(y,\theta)$ of the entire state sequence $Z=(Z_{1},\dots,Z_{T})$ given $Y=y$.

$$
	\hat{Z}_{\text{MAP}}(y,\theta)=\arg\max_{z} p_{Z|Y}(z|y,\theta)=\arg\max_{z}p_{\text{cd}}(y,z\mid \theta).
\tag{42}
$$

We use dynamic programming and define the variables

$$
	\delta_{i}(t) \triangleq \max_{(z_{1},\dots,z_{(t-1)})} P(Z_{1}=z_{1},\dots,Z_{t-1}=z_{t-1},Z_{t}=i,Y_{1}=y_{1},\dots,Y_{t}=y_{t}\mid \theta).
\tag{43}
$$

We leave the derivation of the recurrence to the reader.

**Algorithm (Viterbi).** Compute the $\delta$'s and associated back pointers by a recursion forward in time:

$$
\begin{aligned}
\text{(initial condition)}\quad &\delta_i(1)=\pi(i)b_{i y_1}\\
\text{(recursive step)}\quad &\delta_j(t)=\max_i\{\delta_i(t-1)a_{ij}b_{j y_t}\}\\
\text{(storage of back pointers)}\quad &\phi_j(t)\triangleq\arg\max_i\{\delta_i(t-1)a_{ij}b_{j y_t}\}.
\end{aligned}
\tag{44}
$$

Then $z^* = \hat{Z}_{\text{MAP}}(y,\theta)$ satisfies $p_{\text{cd}}(y, z^*\mid \theta)=\max_{i}\delta_{i}(T)$, and $z^*$ is given by tracking backward in time:

$$
z_T^*=\arg\max_i \delta_i(T)
\qquad\text{and}\qquad
z_{t-1}^*=\phi_{z_t^*}(t)\quad\text{for }2\le t\le T.
\tag{45}
$$

### The Baum-Welch algorithm, or EM algorithm for HMMs

The parameter to be estimated is $\theta = (\pi, A, B)$. The complete data consist of $(Y, Z)$, and the observed data are $Y$. The initial parameter $\theta^{(0)}=(\pi^{(0)}, A^{(0)}, B^{(0)})$ should have all entries strictly positive.

To compute $Q(\theta \mid \theta^{(k)})$, taking logarithms in the expression (27) for the pmf of the complete data yields

$$
	\log p_{\text{cd}}(y,z \mid \theta) = \log \pi_{z_{1}} + \sum_{t=1}^{T-1} \log a_{z_{t}z_{t+1}} + \sum_{t=1}^T \log b_{z_{t}y_{t}}.
\tag{46}
$$

Taking the expectation yields

$$
	\begin{aligned}
	Q(\theta \mid \theta^{(k)}) &= \mathbb{E}\left[ \log p_{\text{cd}}(y, Z \mid \theta) \middle| y, \theta^{(k)} \right] \\
	&= \sum_{i \in \mathcal{S}} \gamma_{i}(1) \log \pi_{i} + \sum_{t=1}^{T-1} \sum_{i,j \in \mathcal{S}} \xi_{ij}(t) \log a_{ij} \\
	&\quad{}+ \sum_{t=1}^T \sum_{i \in \mathcal{S}} \gamma_{i}(t) \log b_{i y_{t}}.
	\end{aligned}
\tag{47}
$$

where the variables $\gamma_{i}(t)$ and $\xi_{ij}(t)$ are defined using the model with parameter $\theta^{(k)}$ according to (39) and (41).

Rewrite $Q(\theta \mid \theta^{(k)})$ for easier optimization:

$$
\begin{aligned}
Q(\theta \mid \theta^{(k)}) &= \sum_{i \in \mathcal{S}} \gamma_{i}(1) \log \pi_{i} \\
&\quad{}+ \sum_{i,j} \left( \sum_{t=1}^{T-1} \xi_{ij}(t) \right) \log a_{ij} \\
&\quad{}+ \sum_{i \in \mathcal{S}} \sum_{\ell \in \mathcal{O}} \left( \sum_{t=1}^T \gamma_{i}(t) \mathbf{1} \{y_{t} = \ell\} \right) \log b_{i\ell }.
\end{aligned}
\tag{48}
$$

We skip the exact derivation. The maximization step of the EM algorithm gives:

$$
\begin{align}
\pi_i^{(k+1)} &= \gamma_i(1), \tag{49}\\
a_{ij}^{(k+1)} &= \frac{\sum_{t=1}^{T-1}\xi_{ij}(t)}{\sum_{t=1}^{T-1}\gamma_i(t)}, \tag{50}\\
b_{i\ell}^{(k+1)} &= \frac{\sum_{t=1}^{T}\gamma_i(t)\,\mathbf{1}_{\{y_t=\ell\}}}{\sum_{t=1}^{T}\gamma_i(t)}. \tag{51}
\end{align}
$$

### Example: Part-of-speech tagging

Part-of-speech (POS) tagging assigns a grammatical category to each token in a sentence, using its context. In the lecture's sentence, *raises* is a verb and *interest* is a noun. In the HMM, $Y_t$ is the observed word and $Z_t$ is its tag.

<figure>
  <img src="../assets/pos-tagging-hmm.svg" alt="Tags and words for the sentence Fed raises interest rates 0.5 percent">
</figure>

Given a trained model, Viterbi finds the most likely tag sequence for a new sentence. An HMM can be trained on untagged text and then used to generate text.

#### Supervised training

When every training word has a known tag, the maximum likelihood estimates are normalized counts. Let $M$ be the number of sentences, and let $\#$ count occurrences across the corpus. Count transitions only within a sentence.

$$
\begin{aligned}
\hat\pi_i
&=\frac{\#\{\text{sentences starting with tag }i\}}{M},\\
\hat a_{ij}
&=\frac{\#\{\text{transitions }i\to j\}}{\#\{\text{transitions out of }i\}},\\
\hat b_{iw}
&=\frac{\#\{\text{tokens with word }w\text{ and tag }i\}}{\#\{\text{tokens with tag }i\}}.
\end{aligned}
\tag{52}
$$

#### Unsupervised training

With only the words observed, the likelihood of a sentence sums over all possible tag sequences. For a sentence of length $T$, the objective is

$$
\max_\theta \log p(y_{1:T}\mid\theta)
=\max_\theta \log\sum_{z_{1:T}\in\mathcal S^T}p_{\mathrm{cd}}(y_{1:T},z_{1:T}\mid\theta).
\tag{53}
$$

For independent sentences, add their log likelihoods. **The Baum-Welch algorithm is EM for this objective.**

## References

- Bruce Hajek. [*Random Processes for Engineers*](https://hajek.ece.illinois.edu/ECE534Notes.html). Cambridge University Press, 2015. Chapter 5, “Inference for Markov models.”
- Kaare Brandt Petersen and Michael Syskind Pedersen. [*The Matrix Cookbook*](https://course.ccs.neu.edu/cs6220f17/ssl/maths/petersen-pederson-matrix-cookbook.pdf). Version: November 15, 2012. Section 2, “Derivatives.”
- [Tianxing He](https://cloudygoose.github.io/). *Natural Language Processing, Part I*. Lecture slides, Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University.
