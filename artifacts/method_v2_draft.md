# Method 章重写草案：VA2（Causal-Decomposed VA Compound）

**状态：草案（2026-08-07），待 MW/LIBERO VA2 数字落地后并入论文 §3**

## 3. Method

We build on the frozen-cache VA compound (Sec. 3.1 of v1) and upgrade it
with four mechanisms that (a) make the language contract causally load-bearing,
(b) protect perceptual evidence from policy intent, (c) give the action→vision
backward path a predictive meaning, and (d) deepen condition injection in the
action head.  All upgrades are optional configuration flags with defaults that
reproduce the v1 behavior exactly; the paper reports both.

### 3.1 Frozen language contract (unchanged)

The instruction is encoded once by frozen Qwen-3.5-2B into per-layer
K/V anchors; action gradients cannot reach language parameters
(parameter-erosion immunity, verified in Sec. 5.5 via the C1/C2 control).

### 3.2 Causal-decomposed memory (memory_split)

The v1 policy maintained one per-layer recurrent state (a snapshot of the
visual stream) that — because the visual stream reads action and language
tokens — is a *multimodal workspace* rather than pure evidence.  This lets
policy intent contaminate the model's belief about the world: after planning
"placing the cup", the memory can carry "placing is happening" even if
execution failed (confirmation bias).  We decompose the recurrent state into

- **Protected evidence memory** $E_t$: updated only by the current frozen
  V-JEPA features $V_t$, the robot state $S_t$ (proprio + previous executed
  action) and the previous evidence, through a gated cross-attention update
  $E_t = (1-g_t) E_{t-1} + g_t \tilde E_t$, with
  $g_t = \sigma(W_g[E_{t-1}, \tilde E_t, \tilde E_t - E_{t-1}])$.  Inside the
  VA attention stack $E_t$ appears **only as a key/value source**: no query
  group (in particular not the action stream) can write into it — protection
  by construction, not by regularization.
- **Task workspace** $T_t$: a small set of tokens that track the dynamic
  reading of the contract — role bindings, satisfied constraints, progress —
  initialized once per episode from the language anchors
  ($T_0 = \mathrm{TaskResampler}(K_L)$) and updated as a read/write stream
  inside the stack with a gated residual $T_t = T_{t-1} + g^T \odot (T^N - T_{t-1})$.

The final action condition becomes $C_t = \mathrm{LN}(A_t + P_T T_t)$:
the flow head is conditioned on the *current task interpretation*, not only
on the static contract.

### 3.3 Sequential action–vision–action coupling (sequential_coupling)

The v1 layer computes visual and action updates from a single joint
attention pass: within one layer, V and A read each other's *previous* state
(synchronous co-attention; multi-layer iteration gives implicit coupling).
Every $k$-th layer (we use $k=2$) instead runs the explicit three-step
cascade proposed by the causal reading of the backward path:

1. **Action proposal** $A^{\frac12} = A + \mathrm{Attn}_A(A \to [V, E, T, L, S])$;
2. **Action-conditioned reorganization** $V', T'$ read $[V, E, A^{\frac12}, T, L, S]$;
3. **Action correction** $A'$ reads $[V', E, A^{\frac12}, T', L, S]$.

This makes "vision interprets the action hypothesis" a first-class
computational step instead of an emergent property.

### 3.4 Future-latent prediction (future_predict)

To give the action→vision backward path physical meaning — predicting what
the executed action will *cause* — a lightweight head maps
$(E_t, T_t, C_t)$ to the mean of the next decision's frozen V-JEPA features,
under a stop-gradient target (the features are precomputed by the frozen
encoder; no future information enters the policy input path):

$$\mathcal L_{\mathrm{future}} = 1 - \cos\!\big(P_\psi(E_t,T_t,C_t),\, \overline{V_{t+1}}\big).$$

This is a regularizer, not a world model: it only requires $C_t$ to carry
predictive information about executed-state change (ablated in Sec. 5.x).
It is kept at a small weight (0.1) and does not gate the action loss.

### 3.5 Shared-source counterfactual flow supervision (L_pair v2)

[Critical fix] The previous pair loss compared flow-head velocities under
*same noise, same τ* but different instructions.  With random τ this is
ill-posed: the interpolated inputs $x_\tau^i \ne x_\tau^j$ already differ by
$\tau(a_i - a_j)$, so a language-blind head can satisfy the delta target.
Our loss evaluates both instructions at a **shared probe**
$x = (1-\tau)\epsilon + \tau \bar a_{ij}$, $\bar a_{ij} = (a_i + a_j)/2$,
$\tau \sim U[0, 0.5]$, with linear-FM targets
$u_i = (a_i - x)/(1-\tau)$, $u_j = (a_j - x)/(1-\tau)$.  Because every
non-language input is identical, the velocity difference at the probe is
attributable to the language condition:

$$\mathcal L_{\mathrm{CF}} = \mathrm{Huber}(v_\theta(x,\tau,C_i), u_i) +
\mathrm{Huber}(v_\theta(x,\tau,C_j), u_j) +
\mathrm{Huber}(v_\theta(x,\tau,C_i) - v_\theta(x,\tau,C_j),\, (a_i - a_j)/(1-\tau)).$$

Pairs are genuine same-scene forks: instructions sharing a scene whose first
state satisfies a cosine-gated contract (feature cosine ≥ 0.99, proprio
max-diff ≤ 0.15, previous action exactly zero — the deployment contract at
episode start).  [Data fact: the MetaWorld MT50 episodes are single-goal and
open/close families start from opposite states (measured min first-state
distance 0.64), so no genuine forks exist there; L_pair evidence comes from
LIBERO, and MW uses FM + future only.]

### 3.6 Deeply conditioned flow head (flow_cond=adaln)

The flow head is a DiT-style transformer with AdaLN-Zero modulation and a
per-layer cross-attention over $C_t$; zero-initialized gates guarantee the
training start is an unconditioned flow field that learns its conditioning
gradually.  [Deployment: 8 Euler steps primary, 32-step reference.]

### 3.7 Total loss (IL side)

$$\mathcal L = \mathcal L_{\mathrm{FM}} + \lambda_{\mathrm{CF}} \mathcal L_{\mathrm{CF}} +
\lambda_{\mathrm{future}} \mathcal L_{\mathrm{future}}$$

with $\lambda_{\mathrm{CF}} = 1.0$ and $\lambda_{\mathrm{future}} = 0.1$.
No other auxiliary objectives.  RL (Sec. 5.7) remains standard PPO over the
flow path with frozen backbones.
