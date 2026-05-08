# Glossary

Definitions of every relevant term for the talk on Rathor, Jaurigue, Ziegler & Schumacher (2026), *Prediction performance of random reservoirs with different topology for nonlinear dynamical systems with different number of degrees of freedom*. Cross-references appear in **bold**.

---

## Reservoir computing core

**Reservoir computing (RC).** A machine-learning paradigm for time-series prediction in which a large fixed random recurrent network — the **reservoir** — transforms an input stream into a high-dimensional dynamical state, and only a linear **readout** layer is trained on top. Cheaper than backpropagation-through-time on standard RNNs while remaining competitive on chaotic-prediction benchmarks.

**Echo state network (ESN).** The canonical implementation of reservoir computing. A discrete-time recurrent network with random fixed input weights $W_{\text{in}}$ and random fixed recurrent weights $W$, a tanh nonlinearity, optional **leaky integration**, and a trained linear output matrix $W_{\text{out}}$. Introduced by Jaeger (2001).

**Liquid state machine (LSM).** A reservoir-computing variant introduced contemporaneously by Maass, Natschläger & Markram (2002) for computational neuroscience. Uses spiking neurons rather than tanh units, but the same conceptual move: a fixed random recurrent network feeding a trained linear readout. ESNs and LSMs are usually treated as two flavors of the same RC paradigm.

**Reservoir.** The fixed random recurrent network at the heart of an ESN. Conceptually plays two roles simultaneously: a *nonlinear feature expansion* of the input into a much higher-dimensional space, and a *fading memory* that mixes the input's recent history into the current state.

**Reservoir state $r(k)$ (also $x(n)$).** The vector of neuron activations at discrete time step $k$. Lives in $\mathbb{R}^N$ where $N$ is the reservoir size (here $N = 1024$).

**Reservoir update equation.** The discrete-time dynamics of the reservoir state:
$$r(k+1) = (1-\varepsilon)\,r(k) + \varepsilon \tanh\!\big(W r(k) + W_{\text{in}} u(k+1) + b\big)$$
The first term is the leak from the previous state, the second is the new contribution driven by recurrence, input, and bias.

**Readout / output layer.** The trained part of the ESN. A linear map $y^p(k+1) = W_{\text{out}}\,r(k+1)$ that projects the high-dimensional reservoir state down to the prediction. Trained by **ridge regression**.

**Echo state property (ESP).** The condition that the reservoir state asymptotically forgets its initial condition: for any long enough input history, $r(k)$ depends only on the input, not on $r(0)$. Empirically holds when the **spectral radius** $\rho(W) \lesssim 1$, though saturating tanh can extend this range.

**Washout (initial transient).** The first $k_0$ time steps of a run, during which the reservoir state is dominated by its arbitrary initial condition rather than the input. These steps are discarded before training the readout. The paper uses $k_0 = 500$.

**Open-loop prediction.** Each time step the reservoir is fed the *true* next input from the ground truth and asked to predict the next state. One-step prediction; no error compounding. The setting used throughout this paper.

**Closed-loop prediction (generative mode).** The reservoir's own output is fed back as the next input. Errors compound, so trajectories eventually diverge from ground truth — this is where the **Lyapunov time** matters. Also called *generative mode* when the goal is to produce a free-running trajectory rather than score short-horizon error.

**Output feedback ($W_{\text{fb}}$).** An optional fixed matrix that feeds the previous output $y(n-1)$ back into the reservoir state update:
$$\tilde{x}(n) = \tanh\!\big(W_{\text{in}}[1; u(n)] + W x(n-1) + W_{\text{fb}} y(n-1)\big).$$
Equivalent to looping the predicted output back into the input channel. Increases representational power but introduces stability problems during training — the trained dynamics now depend on the readout, so a half-trained readout can blow up. Not used by the Rathor paper.

**Teacher forcing.** A trick for training ESNs with output feedback. During training, feed the *target* signal $y^{\text{target}}(n-1)$ — not the reservoir's actual prediction — through $W_{\text{fb}}$. This breaks the recurrence between reservoir and readout, turning the learning problem back into vanilla one-shot ridge regression. Standard practice when feedback is needed.

**Dynamics prediction task (DPT).** The paper's umbrella term for predicting the full $M$-component state $y(k) = [y_1, \ldots, y_M]^T$ at the next time step from a (possibly partial) input $u(k)$. A DPT decomposes into $M$ component-wise prediction subtasks, each either **direct** or **cross**.

---

## Tasks: direct vs cross prediction

**Direct prediction.** A subtask where input component $u_d$ is used to predict the *same* output component $y_d^p$ — predicting a variable from its own history.

**Cross prediction.** A subtask where input component $u_d$ is used to predict a *different* output component $y_{d'}^p$, $d \neq d'$. Required whenever the input dimension is smaller than the output dimension ($N_{\text{in}} < N_{\text{out}}$). The reservoir must reconstruct hidden variables from the temporal history of the observed ones — essentially **delay embedding**.

**The paper's central finding.** Cross-prediction errors dominate the total MSE when $N_{\text{in}} < N_{\text{out}}$, and **symmetric** reservoirs do cross prediction better. Direct prediction is largely topology-indifferent.

---

## Reservoir topology

**Topology.** The wiring pattern of the reservoir matrix $W$ — which nodes connect to which, and what weights those connections carry. Independently of *size* and *spectral radius*, topology can in principle reshape what dynamics the reservoir supports.

**Hadamard product ($\odot$).** Element-wise matrix multiplication: $(A \odot W_c)_{ij} = A_{ij} W_{c,ij}$. Used in the paper to decouple connectivity from weights.

**Connection matrix $A$.** Binary adjacency matrix specifying which node-to-node edges exist. $A_{ij} = 1$ if there is an edge $j \to i$, else $0$.

**Weight matrix $W_c$.** Real-valued matrix with entries drawn from $\mathcal{U}([-0.5, 0.5])$ giving a weight to every potential edge.

**Functional reservoir matrix.** $W = A \odot W_c$. The actual recurrent matrix used in the update equation. Symmetry of $A$ and symmetry of $W_c$ are controlled independently — that's the paper's key design knob.

**Reservoir density $D_r$.** Fraction of nonzero entries in $W$. Held fixed at $D_r = 0.008$ (sparse) throughout the paper, motivated by the high compressibility of biological neural correlations.

**Reservoir size $N$.** Number of neurons in the reservoir. Main analyses use $N = 1024$.

### The five topologies studied

**R-A (random asymmetric).** Both $A$ and $W_c$ asymmetric. Edges are unidirectional ($i \to j$ does not imply $j \to i$). The standard ESN baseline.

**RS-A (random symmetrized connections, asymmetric weights).** $A$ is symmetric (every edge bidirectional), but $W_c$ is asymmetric, so the two directions carry different weights.

**RS-S (random symmetrized, symmetric weights).** Both $A$ and $W_c$ symmetric, so $W$ itself is symmetric. Real spectrum, no rotational modes.

**WS-A (Watts–Strogatz asymmetric weights).** Connection pattern from a Watts–Strogatz small-world network with rewiring probability $p = 1$ (so it's effectively random but with a tighter degree distribution); weights asymmetric.

**WS-S (Watts–Strogatz symmetric weights).** Same WS connection pattern, with symmetric weights.

**Watts–Strogatz network.** A network model that interpolates between regular and random graphs by rewiring edges of a ring lattice with probability $p$. At $p = 1$ it is fully random but with a more concentrated degree distribution than a pure Erdős–Rényi graph.

---

## Linear algebra & spectral concepts

**Spectral radius $\rho(W)$.** The largest absolute value among the eigenvalues of $W$. Controls how strongly the reservoir amplifies past states. After construction, $W$ is rescaled to a target $\rho$ (a tuned hyperparameter).

**Symmetric matrix.** $W = W^T$. Has a fully real spectrum and an orthonormal eigenbasis (spectral theorem). All modes are exponential decay or growth — *no oscillation*.

**Asymmetric matrix.** $W \neq W^T$ in general. Eigenvalues come in complex-conjugate pairs, supporting oscillatory / rotational dynamics.

**Girko's circular law.** For a large random matrix with iid entries (mean 0, variance $1/N$), the eigenvalues fill a disk of radius 1 in the complex plane uniformly. Why R-A's spectrum looks like a disk and RS-S's looks like a line.

**Tikhonov regularization parameter $\gamma$.** The penalty coefficient on $\|W_{\text{out}}\|^2$ in **ridge regression**. Suppresses overfitting and damps blow-up in the readout. One of the paper's three tuned hyperparameters.

**Ridge regression.** Linear least squares with an L2 penalty:
$$W_{\text{out}} = Y^{\text{target}} X^T (XX^T + \gamma I)^{-1}$$
The standard, recommended way to train ESN readouts. Stable, one-shot, regularized.

**Design matrix $X$.** The $(1 + N_u + N_x) \times T$ matrix whose columns are the concatenated inputs and reservoir states $[1; u(n); x(n)]$ collected over the training period. The object actually plugged into the ridge-regression formula.

**Moore–Penrose pseudoinverse $X^+$.** A generalized matrix inverse that solves the least-squares problem $W_{\text{out}} = Y^{\text{target}} X^+$ even when $XX^T$ is singular. An alternative to ridge regression with no regularization — high precision but vulnerable to overfitting.

**Memory capacity (MC).** A scalar measure of how much past input an ESN can reconstruct via its readout. Bounded above by reservoir size $N_x$ for linear reservoirs (Jaeger 2002). A tradeoff with **nonlinear capacity**: more nonlinearity, less memory.

**Leaking rate $\varepsilon$ (also $\alpha$).** The mixing coefficient in the leaky-integrator update; an Euler discretization of a continuous-time ODE. Smaller $\varepsilon$ slows reservoir dynamics, lengthens memory; the paper fixes $\varepsilon = 0.7$.

---

## Hyperparameters & training

**Hyperparameter.** A model parameter not learned from data — set by the experimenter or by grid search. In this paper: $\rho$, $\varepsilon$, $\gamma$ (and $\Delta t$, $N$, $D_r$).

**Grid search.** Brute-force hyperparameter optimization over a discrete mesh of $(\rho, \varepsilon, \gamma)$ — here a $16 \times 10 \times 14$ grid, with $\gamma$ on a log scale.

**Ensemble averaging.** Reporting the *median* MSE over many random instantiations of $W$ (here typically 50–1000) to separate genuine topology effects from realization-to-realization noise.

**Reservoir computing time step $\Delta t$.** The sampling interval at which the input is fed to the RC. A multiple of the underlying dynamical system's integration step $\delta t$. Optimized per task; e.g. $\Delta t = 0.05$ for L63.

**Integration time step $\delta t$.** The step size used by the numerical integrator that generates the ground-truth trajectory of the dynamical system (e.g. $\delta t = 0.002$ for L63).

---

## Online learning algorithms

The Rathor paper uses the standard one-shot ridge regression. But the practical-ESN guide describes several alternative *online* readout-training methods that show up in the broader RC literature.

**Online learning.** Updating $W_{\text{out}}$ incrementally as new data arrives, rather than fitting it once on a complete training set. Required when the data-generating process is non-stationary, or when feedback connections are present and the readout has to adapt while running.

**Least Mean Squares (LMS).** A first-order stochastic gradient method: at each step, nudge $W_{\text{out}}$ in the direction that reduces the instantaneous squared error. Simple and cheap, but converges slowly when the eigenvalues of $XX^T$ are spread over many orders of magnitude — exactly the situation in a high-dimensional reservoir.

**Recursive Least Squares (RLS).** An online algorithm that minimizes an exponentially-discounted version of the squared error
$$E(y, y^{\text{target}}, n) = \frac{1}{N_y} \sum_i \sum_{j=1}^{n} \lambda^{n-j} (y_i(j) - y_i^{\text{target}}(j))^2$$
where $\lambda \in (0, 1]$ is a forgetting factor. Insensitive to eigenvalue spread, much faster convergence than LMS, but quadratic in the number of weights and numerically delicate.

**BackPropagation-DeCorrelation (BPDC).** A specialized online RC algorithm (Steil 2004) with linear-time-per-step complexity. Designed for tasks with output feedback. Tracks rapidly changing signals well but has a short effective memory.

**FORCE learning.** An online method (Sussillo & Abbott 2009) that uses RLS to aggressively pull $W_{\text{out}}$ toward the target right from the start, suppressing the reservoir's spontaneous chaotic activity through the feedback loop. Well-suited to building stable neural pattern generators.

---

## Error metrics

**Mean squared error (MSE).** $\frac{1}{k_p - k_t} \sum_{k=k_t+1}^{k_p} \|y(k) - y^p(k)\|^2$. The total error of a DPT, summed over all output components.

**Normalized root mean squared error (NRMSE).** $\sqrt{\text{MSE}_d / \text{Var}_d}$ for a single component $d$. Dimensionless; scales away from the magnitude of the target. Used to compare per-component performance fairly.

**Normalized average relative error (NARE).** A statistical-correlation-based error used in the SF appendix. Tests whether the predicted velocity field reproduces the right correlations rather than the right pointwise values.

**Performance improvement $I_{\text{err}}$.** $|MSE^w - MSE| / MSE^w$, where $MSE^w$ is the worst-performing topology's error on the same DPT. Shows relative gain of a topology over the worst one.

---

## Dynamical systems studied

**Mackey–Glass equation (MG).** A scalar delay-differential equation
$$\dot u = a \frac{u(t-\tau)}{1 + u(t-\tau)^q} - b u(t).$$
With $\tau = 17$, parameters $a=0.2, b=0.1, q=10$, it generates a chaotic 1D time series. Infinite-dimensional state space (delay equation) but $D_{KY} \approx 2.10$. Acts as the legacy benchmark; a **direct prediction** task ($N_{\text{in}} = N_{\text{out}} = 1$).

**Lorenz 63 (L63).** The original three-variable Lorenz model, derived as a low-dimensional Galerkin truncation of two-dimensional Rayleigh–Bénard convection. Standard parameters $\sigma = 10, b = 8/3, r = 28$. Modes $A_1, B_1, B_2$ correspond to convective velocity and temperature-fluctuation amplitudes. $N_{\text{DoF}} = 3$, $\lambda_{\max} \approx 0.91$, $D_{KY} \approx 2.06$. **The course anchor.**

**Lorenz-type 8 (L8).** An eight-mode Galerkin extension of L63 (Gluhovsky–Tong–Agee 2002) that includes shear within the convection layer and conserves total energy and vorticity. $N_{\text{DoF}} = 8$, $\lambda_{\max} \approx 1.48$, $D_{KY} \approx 3.44$.

**Shear flow (SF).** A nine-mode Galerkin model of three-dimensional plane Couette-like shear flow between free-slip walls (Moehlis–Faisst–Eckhardt 2004). The minimal flow unit for studying the self-sustaining process near transition to turbulence. $N_{\text{DoF}} = 9$, $\lambda_{\max} \approx 0.02$, $D_{KY} \approx 6.25$. Highest dimensional, the most "complex chaos" — and the system that turns out to be **topology-indifferent**.

**Lorenz-96 (L96).** *Mentioned but not studied* in the Rathor paper. Edward Lorenz's 1996 toy model for atmospheric circulation:
$$\dot{x}_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F$$
on a periodic ring of $N$ sites, with forcing $F$ (typically $F = 8$ for chaos, $N$ in the range 36–40 for "global circulation"-scale studies). The advection-like quadratic coupling and constant forcing make it a standard benchmark for chaotic spatiotemporal prediction — much higher-dimensional than L63 yet still cheap to integrate. The Rathor authors speculate (Sec V) that their topology-insensitivity finding for SF will extend to L96 because the polynomial degree of the nonlinearity is the same. Often referred to informally as "the big Lorenz weather model."

**Rayleigh–Bénard convection.** Buoyancy-driven flow of a fluid heated from below and cooled from above between two parallel plates. The classical pattern-forming, route-to-chaos system. Source of L63 and L8.

**Galerkin model.** A low-dimensional approximation of a PDE obtained by expanding the solution in a small set of spatial basis functions and projecting the dynamics onto that basis. L63, L8, and SF are all Galerkin truncations.

---

## Chaos & dynamical-systems concepts

**Chaos.** Deterministic dynamics with sensitive dependence on initial conditions: nearby trajectories diverge exponentially.

**Attractor.** The bounded set in state space toward which trajectories asymptote. For L63 it's the famous butterfly.

**Strange attractor.** A chaotic attractor with fractal (non-integer) dimension.

**Lyapunov exponent $\lambda_r$.** The asymptotic exponential rate at which two nearby trajectories diverge along the $r$-th direction. Computed from QR-decomposition of the Jacobian along a trajectory.

**Lyapunov spectrum.** The full ordered set $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_{N_{\text{DoF}}}$. A positive $\lambda_1$ is the standard signature of chaos.

**Maximal Lyapunov exponent $\lambda_{\max}$.** The largest exponent. Sets the **Lyapunov time** $1/\lambda_{\max}$ — the timescale over which prediction errors grow $e$-fold.

**Lyapunov time.** $\sim 1/\lambda_{\max}$. The natural horizon beyond which closed-loop chaotic prediction must fail. For L63 this is $\sim 1.1$ time units.

**Kaplan–Yorke dimension $D_{KY}$.** An estimate of the fractal dimension of the attractor:
$$D_{KY} = s - \frac{1}{\lambda_{s+1}} \sum_{r=1}^{s} \lambda_r$$
where $s$ is the largest integer such that the partial sum of Lyapunov exponents is non-negative. The paper's "complexity axis" — increases L63 → L8 → SF.

**Phase / state space.** The space of all possible system states. L63's is $\mathbb{R}^3$.

**Poincaré section.** A lower-dimensional slice through state space that records where a trajectory punctures it. Turns continuous flows into discrete maps; useful for visualizing prediction drift on chaotic attractors.

**Delay embedding (Takens' theorem).** A theorem stating that the geometry of an attractor can be reconstructed from a single observable's time-delay coordinates $(u(t), u(t-\tau), u(t-2\tau), \ldots)$, provided the delay dimension is large enough. The mathematical reason cross-prediction is possible: a reservoir with memory is implicitly doing a delay embedding.

---

## High-dimensional / kernel concepts

**Cover's theorem (1965).** Points not linearly separable in low dimension typically *are* linearly separable when projected nonlinearly into a sufficiently high-dimensional space. The conceptual reason a *linear* readout on top of the reservoir works.

**Kernel trick.** The same dimensionality-lift idea, applied implicitly via inner products in SVMs and other kernel methods. Reservoir computing is its dynamical-systems cousin.

**Random feature methods.** Approximate kernel methods by drawing a finite random nonlinear feature map. Reservoirs are random feature maps with memory.

**Information processing capacity (IPC).** A scalar measure of how rich the reservoir's nonlinear-and-memory function basis is. The paper's authors tried to use IPC to *explain* the topology effect — and reports honestly that it didn't work, leaving the mechanism as an open question.

---

## Network-science concepts

**Watts–Strogatz model.** A small-world network model: start from a regular ring lattice, rewire each edge with probability $p$. At $p = 1$ the graph is random; the paper uses $p = 1$.

**Small-world network.** A graph with high local clustering yet short average path length. Common in biological and social networks.

**Degree distribution.** Distribution of the number of edges per node. R-A has different *incoming* and *outgoing* degree distributions (asymmetric); the symmetric topologies have a single shared distribution. WS networks have tighter degree distributions than purely random Erdős–Rényi networks.

**Sparsity.** Fraction of zero entries in the connectivity matrix. Sparse reservoirs are computationally cheap and standard ESN practice; the paper sticks to $D_r = 0.008$.

---

## Notation summary

| Symbol | Meaning |
|---|---|
| $N$ | reservoir size (1024) |
| $N_{\text{in}}, N_{\text{out}}$ | input / output dimensions |
| $N_{\text{DoF}}$ | degrees of freedom of the target system |
| $W$ | recurrent reservoir matrix |
| $W_{\text{in}}$ | fixed random input matrix |
| $W_{\text{out}}$ | trained linear readout |
| $W_{\text{fb}}$ | optional output-feedback matrix |
| $A$ | connection (adjacency) matrix |
| $W_c$ | weight matrix |
| $r(k)$ | reservoir state at step $k$ |
| $u(k)$ | input at step $k$ |
| $y(k)$ | ground-truth target |
| $y^p(k)$ | predicted output |
| $b$ | bias vector |
| $\varepsilon$ | leaking rate (= 0.7) |
| $\rho$ | spectral radius of $W$ |
| $\gamma$ | Tikhonov regularization parameter |
| $D_r$ | reservoir density (= 0.008) |
| $\Delta t$ | RC sampling time step |
| $\delta t$ | underlying integration time step |
| $\lambda_{\max}$ | maximal Lyapunov exponent |
| $D_{KY}$ | Kaplan–Yorke dimension |
| $\odot$ | Hadamard (element-wise) product |
