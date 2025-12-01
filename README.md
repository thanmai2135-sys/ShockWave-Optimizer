# ShockWave-Optimizer
This is an Optimization project.

Abstract
This paper introduces a novel optimization algorithm that adapts its learning behaviour based on observed trajectory patterns such as oscillation, stability, and loss-sensitivity. It combines RMSProp-style preconditioning with dynamic learning rates. Our experiments on well-conditioned quadratics, ill-conditioned quadratics, and the Rosenbrock function indicate that our optimizer significantly outperforms Adam, RMSProp, and SGD with momentum in convergence speed and robustness.
1. Introduction and Novel Idea
1.1 Motivation
Traditional optimization algorithms struggle in curved or poorly conditioned regions. Although methods like Adam, RMSProp, and momentum adapt step sizes or accelerate progress, they do not explicitly interpret the oscillations that occur near a minimum. In our experiments, repeated direction flips consistently appeared near optimal regions, indicating that oscillations are not simply instability but useful signals. By detecting these flips and applying midpoint correction, our method leverages oscillatory behavior to guide the trajectory toward the minimum, improving both stability and convergence speed in difficult landscapes.
1.2 Key Innovation
Our optimizer introduces five innovations that collectively form an oscillation-aware, behavior-driven optimization system:
1. RMSProp-Based Preconditioning: We apply preconditioning to the update magnitudes through RMSProp-style second-moment smoothing, in a manner that ensures the oscillation detection and midpoint snapping operate on well-scaled gradients.
2. Turbo Warm-Up: The optimizer applies an amplified update in the first phase when gradients are small or poorly scaled. This warm-up boost allows quick escape from flat regions and fast early convergence.
3. Stability-Guided Learning Rate Growth: When the optimizer detects long stretches of consistent improvement, it gradually increases the learning rate. Hence, it can move faster across smooth regions with no need for external scheduling.
4. Loss-Sensitive Learning Rate Decay: If a step severely increases the loss, the optimizer immediately shrinks the learning rate and discards the update.
5. Oscillation-Aware Midpoint Snapping: We interpret consecutive direction flips as a sign that the minimum is in the vicinity. When such oscillations are detected, the optimizer performs a midpoint snap, pulling the parameters toward the center of the oscillation zone, which often lies closer to the true minimum. 
1.3 Inspiration and Development
The idea emerged from observing that oscillations consistently appear when the optimizer approaches a minimum, particularly in quadratic bowls and Rosenbrock-type landscapes. Rather than treating this oscillation as an undesirable artifact, we hypothesized that the alternating overshoots encode geometric information about the location of the minimum. Early experiments confirmed that the midpoint between oscillating steps frequently lay closer to the optimum than either endpoint. This insight led to the development of our oscillation-aware snapping mechanism, which was later combined with RMSProp preconditioning and adaptive learning-rate control to form the complete algorithm.
2. Algorithm Description
2.1 Mathematical Formulation
Input: θ₀, α, α_acc, βₛ, γ_grow, γ_decay, p, tol, ε, T
Initialize:
s₀ = 0 				// second-moment accumulator
x_prev = None
last_loss = None
good_steps = 0
osc = 0
losses = []
for t=1 to T do:		// Compute loss and gradient
 Loss t = f(θt−1)
 Append lost​ t to losses
gt=∇f(θt−1)
if ∥gt∥ < tol:
    return output 
 // Update preconditioner (RMSProp-like second moment)
st=βsst−1+(1−βs)gt2
// Scale gradient
 gt~=gt/(st1/2+ε)
 // Warm-up accelerated step
  if t<10:
  θttemp=θt−1−α⋅αacc⋅g~t   
else:
θttemp=θt−1−α⋅g~t
// Check loss change (accept/reject step)
     new_loss = f(θttemp)
  if last_loss ≠ None and new_loss > 1.02 × last_loss: // rollback and shrink learning rate
        α=γdecay⋅α 
continue to next iteration
   // Accept step
     xprev=θt−1
     θt=  θttemp 
 last_loss = new_loss
    // Track good steps → grow learning rate
    good_steps += 1
    if good_steps > 20:
	α=γgrow⋅α
	good_steps = 0
// Oscillation detection
     if xprev​ ≠ None and θt−xprev,gt   >0:
	osc += 1
     else:
	osc = 0
// Midpoint snap to reduce oscillation
     if osc ≥ p:
           θt=1/2(θt+xprev)
	osc = 0
end for
Output: θt
2.2 Hyperparameters


2.3 Relationship to Existing Optimizers
Our Optimizer V4 can be viewed as an extension of existing optimizers with the following modifications:
Compared to Adam: Adam uses fixed β₁ and β₂ along with a fixed learning rate, we make the learning rate adaptive based on stability, oscillation and loss sensitivity. oscillation midpoint snapping further adjusts the update direction, when learning rate growth/decay, and snapping are disabled, AdaptiveMomentum-V4 reduces to an Adam-like update.
Compared to SGD+Momentum: Standard momentum uses a fixed β and does not react to trajectory instability. Our algorithm generalizes this by adapting the effective step size, applying loss-based braking, and damping oscillations through midpoint snapping. Setting dynamic adaptation off and keeping α constant yields behavior similar to heavy-ball momentum.
Novel aspects:The oscillation detection, midpoint snapping, warm-up acceleration, dynamic learning-rate growth/decay, and loss-sensitive braking are unique to ShockWave and not present in standard optimizers such as Adam, RMSProp, or SGD with momentum.
3. Experimental Results
3.1 Experimental Setup
We evaluate our optimizer on four benchmark tasks:
Well-conditioned quadratic minimization
Ill-conditioned quadratic minimization
Rosenbrock function optimization
Wine DataSet
All experiments compare our optimizer against:
SGD with momentum (β = 0.9)
Adam (default parameters)
RMSProp (default parameters)
Each experiment is run from the same initial point per seed, and we report the number of iterations required to converge, along with loss-vs-iteration trajectories.
3.2 Quadratic Function
Well-Conditioned
Problem: Optimize a 10-dimensional quadratic with condition number 10.
Results:
Our Optimizer: 117 iterations
Adam: 779 iterations
SGD+Momentum: 1267 iterations
RMSProp: 20000 iterations
Analysis: Our optimizer converges 6x to 17x faster than the baselines. Warm-up acceleration and adaptive LR growth- rapid early descent, while RMSProp smoothing prevents divergence. Adam is solid, but RMSProp stalls from overly cautious updates. 
Ill-Conditioned
Problem: Optimize the same 10-dimensional quadratic but with a condition number of 100.
Results:
Our Optimizer: 686 iterations
Adam: 8250 iterations
SGD+Momentum: 13595 iterations
RMSProp: 20000 iterations
Analysis: Adam and SGD need far more iterations, and RMSProp fails to reach tolerance. Our optimizer manages the steep curvature using midpoint snapping and preconditioning, achieving 10-20× faster convergence than the baseline optimizers.

3.3 Rosenbrock Function
Problem: Optimize the classical Rosenbrock function, which features a narrow curved valley that induces strong zig-zag behaviour.
Results:
Our Optimizer: 9454 iterations
Adam: 12314 iterations
SGD+Momentum: 20000 iterations (did not converge)
RMSProp: 20000 iterations (stalled)
Analysis: The Rosenbrock valley poses a significant challenge for gradient-based methods. SGD+Momentum and RMSProp fail, while Adam progresses slowly. Our optimizer performs the best by using oscillation-aware midpoint snapping.

3.4 Wine Dataset
Problem:Train a simple neural network on the Wine dataset to compare optimizer performance on a real-world classification task.(20 epochs, batch size 16.) 
Results:
Our Optimizer: 100.00% accuracy
Adam: 98.15% accuracy
SGD: 98.15% accuracy
RMSProp: 100.00% accuracy
Analysis: Our optimizer benefits from warm-up accelerations in early steps and RMSProp-style smoothing throughout training.Both achieve perfect test accuracy. Adam and SGD have slightly higher loss in earlier epochs and converge to a slightly lower accuracy.This proves that our optimizer remains competitive on real data, ensuring stability and strong generalization even outside of synthetic benchmark functions.

4. Conclusion
We present an oscillation-aware optimizer that interprets repeated direction flips as a proxy for proximity to a minimum. With midpoint snapping, warm-up acceleration, adaptive learning-rate scaling, and RMSProp-style preconditioning, it converges faster and more stably compared to SGD with momentum, Adam, and RMSProp. Consistent gains on well- and ill-conditioned quadratics as well as the Rosenbrock function suggest that oscillations can inform optimization in curved landscapes.
Limitations remain: tests are restricted to low-dimensional synthetic problems, behavior in noisy or high-dimensional settings is unclear, and the method contains hyperparameters that may require tuning. Future work involves scaling to larger models, studying high-dimensional oscillation patterns, and developing more adaptive or theoretically grounded snapping rules for stochastic and real-world tasks.
Group Contribution: Sidhant and Aadi handled technical development-ideation, coding, implementation, and tuning. Gayathri and Akshitha handled design and presentation—report layout and poster creation.

References
https://www.researchgate.net/publication/337600944_Research_on_Rosenbrock_Function_Optimization_Problem_Based_on_Improved_Differential_Evolution_Algorithm
http://www.cs.toronto.edu/~sajadn/sajad_norouzi/ECE1505.pdf
https://medium.com/@florian_algo/optimization-algorithm-from-sgd-to-adam-50ea22187951



