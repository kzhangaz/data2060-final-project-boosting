# AdaBoost (Adaptive Boosting)
Authors: Wendi Liao, Yicheng Lu, Xuetong Tang, Ke Zhang

## Package Versions
Python: 3.10.1
scikit-learn: 1.5.1
NumPy: 2.0.1
pandas: 2.2.2
seaborn: 0.13.2

## Overview of AdaBoost
AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines multiple weak classifiers to build a strong classifier. It is particularly effective for binary classification problems, but it can also be adapted to multi-class tasks. The main idea is to sequentially train weak learners, typically decision stumps, by focusing on the misclassified samples from previous rounds. In each iteration, misclassified samples are given higher weights to ensure the next classifier focuses more on the challenging cases.

### Advantages
- **Improves accuracy**: By combining multiple weak classifiers, AdaBoost can significantly improve prediction accuracy.
- **Simple to implement**: AdaBoost mainly requires a simple weak learner, like a decision stump.
- **Less parameter tuning**: AdaBoost adapts based on the data, so it doesn’t require extensive parameter tuning.

### Disadvantages
- **Sensitive to noisy data**: AdaBoost focuses on hard-to-classify instances, so it can overfit when faced with noisy data.
- **Needs careful handling for weak classifiers**: If weak classifiers perform worse than random guessing, AdaBoost may degrade.



## Representation

Let $\mathcal{H}$ be the class of base, i.e., not-boosted hypotheses. In $\mathcal{E}$, we collect $T$ number of weak learners (decision stumps in this case). Let $\mathcal{E}(\mathcal{H}, T)$ be the class of ensemble hypotheses built using $T$ elements of $\mathcal{H}$: 

$$ 
\mathcal{E}(\mathcal{H}, T) = \\{ x \mapsto h_S(x) =  \text{sign} \left( \sum_{t=1}^{T} w_t h_t(x) \right) : w \in \mathbb{R}^T, \forall t, h_t \in \mathcal{H} \\}
$$

where:
- $w_t$ represents the weight of each weak learner, not the weight vector.


## Loss Function
AdaBoost minimizes an exponential loss function, which penalizes misclassified samples more heavily:

- Overall loss function is just 0-1 loss:

$$
L_S(h_S) = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}_{[h_S(x_i) \neq y_i]}
$$

- But we redefine the loss on training examples for the $t$-th base hypothesis:

$$
\epsilon_t \overset{\text{def}}{=} L_{D^{(t)}}(h_t) \overset{\text{def}}{=} \sum_{i=1}^{m} D_i^{(t)} \mathbb{1}_{[h_t(x_i) \neq y_i]} \quad \text{where} \quad D^{(t)} \in \mathbb{R}^m
$$


where:
- $m$ is the number of samples
- $y_i$ is the true label for instance $i$.
- $h_t(x_i)$ is the prediction from the $t$-th classifier for instance $i$.
- $D_i^{(t)}$ is the distribution over the m examples in S for the t-th base learner, and sums to 1. The calculation of  $D_i^{(t)}$ is shown below in optimizer section.

## Optimizer
AdaBoost uses a greedy optimization algorithm to determine the optimal weights $\alpha_t$ for each weak learner. The algorithm iterates as follows:
**Input:**
- Training set $S = (x_1, y_1), \dots, (x_m, y_m)$
- Weak learner WL
- Number of rounds $T$

**Initialize** $D^{(1)} = \left( \frac{1}{m}, \dots, \frac{1}{m} \right)$.

**For** $t = 1, \dots, T$:
- Invoke weak learner $h_t = WL(D^{(t)}, S)$
- Compute $\epsilon_t = \sum_{i=1}^{m} D_i^{(t)} \mathbb{1}_{[y_i \neq h_t(x_i)]}$
- Let $w_t = \frac{1}{2} \log \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$
- Update $D_i^{(t+1)} = \frac{D_i^{(t)} \exp(-w_t y_i h_t(x_i))}{\sum_{j=1}^{m} D_j^{(t)} \exp(-w_t y_j h_t(x_j))}$ for all $i = 1, \dots, m$

**Output** the hypothesis $h_S(x) = \text{sign} \left( \sum_{t=1}^{T} w_t h_t(x) \right)$.
