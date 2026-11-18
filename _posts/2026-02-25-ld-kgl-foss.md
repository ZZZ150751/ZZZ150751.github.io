---
layout: post
title: "LD-KGL: What the Biggest Gradient Misses"
date: 2026-02-25
description: "What the two-dimensional Foss example teaches us about guiding neural networks toward singular solutions."
published: true
math: true
tags: [scientific-computing, variational-problems]
---
<div style="
  background-color: var(--global-code-bg-color, #f7f7f7);
  border-left: 4px solid var(--global-theme-color, #b509ac);
  border-radius: 6px;
  padding: 1.2rem 1.5rem;
  margin: 1.5rem 0 2rem;
  font-family: Georgia, 'Times New Roman', serif;
  line-height: 1.7;
">

  <p style="font-size: 1.15rem; font-weight: 700; margin-top: 0;">
    Overview
  </p>

  <p>
    The two-dimensional Foss's example is a variational problem with the
    Lavrentiev phenomenon. Its solution has a boundary singularity whose
    strength varies with position, while the guiding term in
    knowledge-guided learning (KGL) focuses on the largest gradient over
    the domain.
  </p>

  <p style="margin-bottom: 0;">
    In this post, we introduce **Location-Dependent Knowledge-Guided Learning (LD-KGL)**, which incorporates the exponent of
    the singularity into a weighted guiding term. We also apply a pointwise
    logarithmic transformation to the energy density to improve numerical
    stability. In our experiment, LD-KGL achieves a relative
    <i>L</i><sup>2</sup> error of 0.002015, compared with 0.046215 for
    KGL—a reduction of approximately 95.6%.
  </p>

</div>

## Variational Problems and the Lavrentiev Phenomenon

Let $\Omega\subset\mathbb R^m$ be a bounded domain. Consider the variational problem

$$
\min_{u\in\mathcal A} I[u],
\qquad
I[u]=\int_\Omega L\bigl(x,u(x),\nabla u(x)\bigr)\,\mathrm dx,
$$

where $\mathcal A$ is the class of admissible functions satisfying the prescribed boundary conditions.

Let $\mathcal A_1\subset\mathcal A_0$ be a dense subset of the admissible class $\mathcal A_0$. The **Lavrentiev phenomenon** occurs when [2]

$$
\inf_{u\in\mathcal A_0} I[u]
<
\inf_{u\in\mathcal A_1} I[u].
$$

The difference between these infima is the **Lavrentiev gap**.

## The Foss's Example

The Foss's example is a classical two-dimensional variational problem that exhibits the Lavrentiev phenomenon. The strength of the solution's singularity along the boundary $x=0$ varies with $y$ [3].  The domain is

$$
\Omega=[0,1]\times[1.5,2.5].
$$

The energy functional is

$$
\begin{aligned}
I_{\mathrm{Foss}}[u]
={}&\int_{1.5}^{2.5}\int_0^1
C_0\left(\frac{y}{y-1}\right)^{14}
|u(x,y)|^{\frac{14-3y}{y-1}}\\
&\quad{}\times
\left(|u(x,y)|^{\frac{y}{y-1}}-x\right)^2
|\partial_x u(x,y)|^{14}\,\mathrm dx\,\mathrm dy,
\end{aligned}
$$

where $C_0=66(13/14)^{14}$. Below, we write $I$ for this functional and $\rho_u(x,y)$ for its integrand.

The analytical solution is

$$
u_*(x,y)=x^{(y-1)/y}.
$$

The Dirichlet boundary conditions are imposed at the two ends in the $x$ direction:

$$
u(0,y)=0,\qquad u(1,y)=1,\qquad y\in[1.5,2.5].
$$

Its derivative in the $x$ direction is

$$
\partial_x u_*(x,y)=\frac{y-1}{y}x^{-1/y}.
$$

This derivative diverges as $x \to 0$, with a rate that depends on $y$. The $y$-derivative is bounded. Thus $u_*$ belongs to $W^{1,1}(\Omega)$ but not to $W^{1,\infty}(\Omega)$.

Let $\mathcal A_0$ and $\mathcal A_1$ be the admissible classes in $W^{1,1}(\Omega)$ and $W^{1,\infty}(\Omega)$, respectively, with the boundary conditions above. Then the infima satisfy [3]

$$
\inf_{u\in\mathcal A_0} I[u]=0
<
\inf_{u\in\mathcal A_1} I[u]=1.
$$

This gap poses a difficulty for standard conforming finite element methods: their trial functions lie in $\mathcal A_1$, so even on a finer mesh, the original energy cannot fall below $1$ and approach the exact solution's energy of zero.

## Deep Ritz and knowledge-guided learning

Neural networks offer a way to approximate singular solutions without relying on a fixed mesh. The Deep Ritz method represents the trial function by a neural network $$u_\theta$$ and trains it by minimizing a sampled approximation to the energy [1]. The exact solution satisfies $$\partial_x u_*\to+\infty$$ as $$x\to0$$. When the residual is still large, the factor $$\lvert \partial_x u_\theta\rvert^{14}$$ makes steep gradients costly, so standard Deep Ritz tends to favor smoother approximations.

To solve this problem, Wang, Xu, and Zou [3] proposed knowledge-guided learning (KGL), which alternates between minimizing the original variational energy and minimizing a guiding loss based on the magnitudes of the network's spatial derivatives. The guiding step encourages larger derivatives, helping the network approximate the singular solution.

## What the largest gradient leaves out

KGL improves the approximation in the Manià example [3]. In our experiments on the two-dimensional Foss's problem, however, it produced results close to those of standard Deep Ritz. The two error maps below show the comparison.

<div style="display:flex;flex-wrap:wrap;gap:0.75rem;margin:1rem 0;">
  <figure style="flex:1 1 250px;min-width:0;margin:0;">
    <img src="{{ '/assets/img/ld-kgl/foss-deep-ritz-error.png' | relative_url }}" alt="Signed error map for standard Deep Ritz on the Foss's example" width="640" height="480" loading="lazy" style="display:block;width:100%;height:auto;">
    <figcaption>Standard Deep Ritz</figcaption>
  </figure>
  <figure style="flex:1 1 250px;min-width:0;margin:0;">
    <img src="{{ '/assets/img/ld-kgl/foss-kgl-error.png' | relative_url }}" alt="Signed error map for KGL on the Foss's example" width="640" height="480" loading="lazy" style="display:block;width:100%;height:auto;">
    <figcaption>Knowledge-guided learning (KGL)</figcaption>
  </figure>
</div>

*Signed error on the Foss's example. The horizontal and vertical axes indicate test grid point indices.*

The guiding term measures the largest gradient magnitude over the domain. A large gradient in a small region can dominate this quantity, even if the $x$-derivative remains too small near other parts of the left boundary.

The slice plots compare the KGL solution with the analytical solution. Choose **Along x** to hold $y$ fixed, or **Along y** to hold $x$ fixed. Each slider step changes the fixed coordinate by $0.1$.

<iframe
  src="{{ '/assets/img/ld-kgl/foss-slices/index.html' | relative_url }}"
  title="Interactive Foss solution slices along x and y"
  loading="lazy"
  width="100%"
  height="550"
  style="display:block;width:100%;border:0;margin:0.75rem 0;"
></iframe>

Along the $y$-direction, the KGL solution differs noticeably from the analytical solution at many fixed values of $x$.

To better understand the behavior near the left boundary, we can look at the $x$-derivative of the analytical solution. At a fixed height $y$, it grows like $x^{-1/y}$. Consider two horizontal slices at $y=1.5$ and $y=2.5$. As $x$ decreases from $10^{-1}$ to $10^{-4}$, the derivative grows by a factor of **100** on the first slice and about **15.8** on the second. Thus, the derivative grows at different rates along the left boundary.

This is the motivation for LD-KGL: **we use the exponent $-1/y$ to give stronger guidance where the derivative is expected to grow faster.**

## How LD-KGL uses the exponent

### An exponent-aware guiding term

We first construct a weight that depends on the position $(x,y)$:

$$
w(x,y)=\min\left\{
\frac{1}{(x+\varepsilon)^{1/y}},\,w_{\max}
\right\}.
$$

Here, $\varepsilon$ regularizes the weight near $x=0$, and
$w_{\max}$ prevents it from becoming too large. 

Using this weight, we define the guiding objective as the negative
$p$-th power of a weighted $L^p$ seminorm of the $x$-derivative:

$$
G(u)=-\lambda\int_{\Omega}
\bigl(w(x,y)|\partial_xu(x,y)|\bigr)^p\,\mathrm dx\,\mathrm dy.
$$

Minimizing $G$ encourages larger $x$-derivatives, with stronger
guidance at points assigned larger weights. Among the values tested in
our experiments, $p=4$ gave the best results. We use
$\varepsilon=10^{-4}$, $w_{\max}=50$, and $\lambda=0.015$.

### Stabilizing the energy

The high-order term $|\partial_xu|^{14}$ in the Foss energy can
cause numerical instability during training. To reduce its effect, we
apply a logarithmic transformation to the energy density at each point:

$$
\widetilde{\rho}_u(x,y)=\log\bigl(1+\rho_u(x,y)\bigr).
$$

The stabilized energy is

$$
\widetilde I[u]
=\int_{\Omega}\log\bigl(1+\rho_u(x,y)\bigr)\,\mathrm dx\,\mathrm dy.
$$

During training, we average the transformed energy density over the
sampled points. This compresses large energy values and reduces the
influence of high-energy points on the update.

The core PyTorch computations of LD-KGL
are shown below. 

```python
def foss_density(u, ux, x, y):
    c0 = 66 * (13 / 14) ** 14
    return (
        c0 * (y / (y - 1)) ** 14
        * u.abs() ** ((14 - 3 * y) / (y - 1))
        * (u.abs() ** (y / (y - 1)) - x).square()
        * ux.abs() ** 14
    )


def energy_loss(u, ux, x, y):
    return torch.log1p(foss_density(u, ux, x, y)).mean()


def guide_loss(ux, x, y, p=4, eps=1e-4, w_max=50.0, lam=0.015):
    weight = (x + eps).pow(-1.0 / y).clamp(max=w_max)
    return -lam * (weight * ux.abs()).pow(p).mean()
```

## Numerical results

We compare LD-KGL with standard Deep Ritz and KGL on the
two-dimensional Foss's example. The table reports the relative
$L^2$ error of each method.

| Method | Relative $L^2$ error |
|:---|---:|
| Deep Ritz | 0.047465 |
| KGL | 0.046215 |
| LD-KGL | **0.002015** |

In this experiment, KGL gives an error close to that of Deep Ritz.
LD-KGL reduces the relative $L^2$ error by approximately 95.6%
compared with KGL.

The figure below shows the relative error of LD-KGL across the domain.
The largest error appears near the left boundary, where the analytical
solution has a singular derivative.

<figure style="margin:1rem auto;text-align:center;">
  <img src="{{ '/assets/img/ld-kgl/foss_LD_KGL_error.png' | relative_url }}"
       alt="Relative error of LD-KGL on the Foss's example"
       style="max-width:100%;height:auto;">
  <figcaption>Relative error of LD-KGL on the Foss's example.</figcaption>
</figure>

The following heatmaps compare the analytical solution with the solution
obtained by LD-KGL.

<div style="display:flex;flex-wrap:wrap;gap:0.75rem;margin:1rem 0;">
  <figure style="flex:1 1 250px;min-width:0;margin:0;">
    <img src="{{ '/assets/img/ld-kgl/Exact_Solution.png' | relative_url }}"
         alt="Analytical solution of the Foss's example"
         style="display:block;width:100%;height:auto;">
    <figcaption>Analytical solution</figcaption>
  </figure>

  <figure style="flex:1 1 250px;min-width:0;margin:0;">
    <img src="{{ '/assets/img/ld-kgl/LD_KGL_Solution.png' | relative_url }}"
         alt="LD-KGL solution of the Foss's example"
         style="display:block;width:100%;height:auto;">
    <figcaption>LD-KGL solution</figcaption>
  </figure>
</div>


## Cite this post

If you find this post useful, you can cite it as follows:

```bibtex
@misc{zhang2026ldkgl,
  author = {Zitong Zhang and Qixuan Wang and Hongbin Shi},
  title  = {LD-KGL: What the Biggest Gradient Misses},
  year   = {2026},
  month  = {February},
  note   = {Blog post},
  url    = {https://zzz150751.github.io/blog/2026/ld-kgl-foss}
}
```
## References

1. Weinan E and Bing Yu. *The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm for Solving Variational Problems*. Communications in Mathematics and Statistics, 6(1): 1–12, 2018. [Paper](https://doi.org/10.1007/s40304-018-0127-z)
2. Mikil Foss. *Examples of the Lavrentiev Phenomenon with Continuous Sobolev Exponent Dependence*. Journal of Convex Analysis, 10(2): 445–464, 2003. [Paper](https://doi.org/10.68381/jca1027)
3. Dong Wang, Xianmin Xu, and Boyi Zou. *Deep Learning Methods for Singular Variational Problems with Lavrentiev Phenomena*.  [Preprint](https://lsec.cc.ac.cn/~xmxu/publication/Lavrentiev_Phenomenon.pdf)
