# ShockWave-Optimizer
This is an Optimization project.
This paper introduces a novel optimization algorithm that adapts its learning behaviour based on observed trajectory patterns such as oscillation, stability, and loss-sensitivity. It combines RMSProp-style preconditioning with dynamic learning rates. Our experiments on well-conditioned quadratics, ill-conditioned quadratics, and the Rosenbrock function indicate that our optimizer significantly outperforms Adam, RMSProp, and SGD with momentum in convergence speed and robustness.

1.1 Motivation
Traditional optimization algorithms struggle in curved or poorly conditioned regions. Although methods like Adam, RMSProp, and momentum adapt step sizes or accelerate progress, they do not explicitly interpret the oscillations that occur near a minimum. In our experiments, repeated direction flips consistently appeared near optimal regions, indicating that oscillations are not simply instability but useful signals. By detecting these flips and applying midpoint correction, our method leverages oscillatory behavior to guide the trajectory toward the minimum, improving both stability and convergence speed in difficult landscapes.
1.2 Key Innovation
Our optimizer introduces five innovations that collectively form an oscillation-aware, behavior-driven optimization system:
1. RMSProp-Based Preconditioning: We apply preconditioning to the update magnitudes through RMSProp-style second-moment smoothing, in a manner that ensures the oscillation detection and midpoint snapping operate on well-scaled gradients.
2. Turbo Warm-Up: The optimizer applies an amplified update in the first phase when gradients are small or poorly scaled. This warm-up boost allows quick escape from flat regions and fast early convergence.
3. Stability-Guided Learning Rate Growth: When the optimizer detects long stretches of consistent improvement, it gradually increases the learning rate. Hence, it can move faster across smooth regions with no need for external scheduling.
4. Loss-Sensitive Learning Rate Decay: If a step severely increases the loss, the optimizer immediately shrinks the learning rate and discards the update.
5. Oscillation-Aware Midpoint Snapping: We interpret consecutive direction flips as a sign that the minimum is in the vicinity. When such oscillations are detected, the optimizer performs a midpoint snap, pulling the parameters toward

6. Algorithm Description
2.1 Mathematical Formulation
Input: θ₀, α, α_acc, β􀀀, γ_grow, γ_decay, p, tol, ε, T
Initialize:
s₀ = 0 // second-moment accumulator x_prev = None last_loss = None good_steps = 0 osc = 0 losses = []
for t=1 to T do: // Compute loss and gradient Loss t = f(θt−1) Append lost t to losses gt=∇f(θt−1)
if ∥gt∥ < tol: return output
// Update preconditioner (RMSProp-like second moment)
st=βsst−1+(1−βs)gt2
// Scale gradient gt~=gt/(st1/2+ε)
// Warm-up accelerated step if t<10: θttemp=θt−1−α⋅αacc⋅g~t
else: θttemp=θt−1−α⋅g~t
// Check loss change (accept/reject step) new_loss = f(θttemp)
if last_loss ≠ None and new_loss > 1.02 × last_loss: // rollback and shrink learning rate α=γdecay⋅α
continue to next iteration
// Accept step xprev=θt−1 θt= θttemp
last_loss = new_loss
// Track good steps → grow learning rate
good_steps += 1 if good_steps > 20: α=γgrow⋅α good_steps = 0
// Oscillation detection
if xprev ≠ None and θt−xprev,gt >0: osc += 1 else: osc = 0
// Midpoint snap to reduce oscillation if osc ≥ p: θt=1/2(θt+xprev) osc = 0
end for
Output: θt
