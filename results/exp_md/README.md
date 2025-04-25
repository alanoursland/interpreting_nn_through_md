# Results: Learning Mahalanobis Distance with Neural Networks

## Overview

This experiment demonstrates the key theoretical claims of the paper "Interpreting Neural Networks through Mahalanobis Distance" by training small neural networks to approximate the Mahalanobis distance for synthetic Gaussian data. It evaluates whether specific architectural choices (Abs, ReLU, Sigmoid) lead to learned representations that align with the theory.

The key outcomes to look for in these results are:

* Whether the learned model outputs closely match true Mahalanobis distances (MSE loss).
* Whether the model weights reflect the underlying Gaussian structure (eigenvectors, whitening).
* Whether the architectural choices affect alignment with theoretical expectations (e.g., ReLU pairs).

---

## Experiment Description

The core task is to train neural network models to predict the Mahalanobis distance $D_M(x)$ for input data points $x$ sampled from a predefined Gaussian distribution $N(\mu, \Sigma)$. This validation was conducted using synthetic Gaussian data generated in multiple dimensions ($d=2$, $d=10$, and $d=100$) to assess the scalability and robustness of the findings.

* **Input:** $d$-dimensional data vectors $x \sim N(\mu, \Sigma)$ (where $d \in \{2, 10, 100\}$).
* **Target Output:** The true Mahalanobis distance $D_M(x) = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}$.
* **Goal:** To train models such that their output $f(x)$ approximates $D_M(x)$, typically by minimizing the Mean Squared Error (MSE) loss $E[(f(x) - D_M(x))^2]$, and to analyze whether the learned parameters align with the theoretical interpretation explored in this project.

---

## Model Architectures

Three model architectures (`MD_Abs`, `MD_ReLU`, `MD_Sigmoid`) are implemented and compared, each utilizing a different activation function within a structure designed to compute a distance metric. All models take the $d$-dimensional input $x$ and output a scalar distance approximation.

1.  **Absolute Value Model (`MD_Abs`)**
    * **Architecture:** `Linear(d -> d)` -> `Abs` -> `Square` -> `Sum` -> `Sqrt`
    * **Justification:** This architecture directly reflects the core theoretical connection proposed in Section 3.3 of the paper. The theory suggests that a linear transformation $Wx+b$ followed by an absolute value activation can model the scaled projection of the data onto a principal component direction, $|w_i^T x + b_i| \approx |\lambda_i^{-1/2} v_i^T (x-\mu)|$. Squaring and summing these components computes the squared $l_2$ norm, equivalent to the squared Mahalanobis distance. The final square root yields the distance.
    * **Note on Implementation:** While the `Abs` activation layer is included here to directly mirror the absolute value used in the theoretical derivation, it is mathematically redundant in this specific implementation because the subsequent squaring operation ($z^2 = |z|^2$) inherently removes any sign information. The explicit `Abs` layer is retained primarily for conceptual clarity and thematic comparison with the `MD_ReLU` architecture.

2.  **ReLU-based Model (`MD_ReLU`)**
    * **Architecture:** `Linear(d -> 2d)` -> `ReLU` -> `Square` -> `Sum` -> `Sqrt`
    * **Justification:** This model tests the hypothesis from Section 4.2 that ReLU activations can be functionally equivalent to Abs activations for this task. It leverages the identity $Abs(z) = ReLU(z) + ReLU(-z)$. By using $2d$ neurons in the linear layer, the model can potentially learn pairs of neurons where one captures the positive part $ReLU(w_i^T x + b_i)$ and the other captures the negative part $ReLU(-w_i^T x - b'_i)$ of the deviation along axis $i$. Summing the squares of all $2d$ ReLU outputs aims to approximate the sum of squares of the absolute values, thereby computing the squared Mahalanobis distance.

3.  **Sigmoid Model (`MD_Sigmoid`)**
    * **Architecture:** `Linear(d -> 2d)` -> `Sigmoid` -> `Square` -> `Sum` -> `Sqrt`
    * **Justification:** This model is included as a baseline using a traditional activation function. To maintain structural parallelism with the `MD_ReLU` model for comparison, it also uses $2d$ nodes in its linear layer, notionally attempting to capture deviations on both positive and negative sides similar to how ReLU pairs aim to mimic the absolute value function. However, unlike Abs and ReLU, the Sigmoid activation lacks a direct theoretical basis for representing distance components or effectively mimicking the absolute value function required by the subsequent `Square -> Sum -> Sqrt` operations (which derive their meaning from the Mahalanobis distance calculation). Due to Sigmoid's saturating nature and output range (0 to 1), **this architecture is not expected to perform well** in accurately approximating the Mahalanobis distance. Its inclusion primarily serves to contrast the performance of the theoretically motivated architectures (Abs, ReLU) against a standard activation function placed within the same distance-approximating computational graph.

Absolutely — here's a polished and mathematically accurate version of your explanation, keeping your intent and structure but refining the math description and clarity:

### **Ground Truth Model**

We construct a **ground truth model** by explicitly initializing the linear layer of `MD_Abs` using the known eigendecomposition of the synthetic Gaussian distribution. This serves as a direct implementation of the Mahalanobis distance from first principles.

Specifically:

- The **weight matrix** is set as  
  \[
  W = \Lambda^{-1/2} V^\top
  \]  
  where \( V \in \mathbb{R}^{d \times d} \) contains the eigenvectors of the covariance matrix \( \Sigma \) as columns, and \( \Lambda \in \mathbb{R}^{d \times d} \) is a diagonal matrix of corresponding eigenvalues. This transformation whitens the data by projecting it onto the scaled principal component axes.

- The **bias vector** is set to  
  \[
  b = -W \mu
  \]  
  ensuring that the transformed space is centered around the mean \( \mu \), and each linear hyperplane passes exactly through the center of the Gaussian distribution.

This model is **not trained**; it is constructed analytically and used as a reference to:

- **Validate** analysis metrics such as whitening accuracy, Mahalanobis error, and eigenvector alignment.
- **Benchmark** trained models against the theoretical ideal they are designed to approximate.
- **Visualize** exact Mahalanobis directions and decision boundaries in 2D, providing intuitive insight into feature space geometry.

By analytically encoding the Mahalanobis distance within the `MD_Abs` architecture, the ground truth model serves as a constructive proof-of-concept for the proposed interpretability framework.

---

## Training and Test Data

The data used in this experiment was synthetically generated to provide a controlled environment for evaluating the models' ability to learn the Mahalanobis distance across different dimensionalities ($d=2, 10, 100$).

For each dimension $d$, a specific multivariate Gaussian distribution, $N(\mu, \Sigma)$, was defined. The parameters were generated once for each dimension using the `create_gaussian` function:

* **Mean Vector ($\mu$):** The mean vector $\mu \in \mathbb{R}^d$ was generated by sampling each element independently from a uniform distribution over the range [-10, 10].
* **Covariance Matrix ($\Sigma$):** The $d \times d$ positive semi-definite covariance matrix $\Sigma$ was constructed as follows:
    1.  A random matrix $A$ of shape $(d, d)$ was created with entries drawn from a standard normal distribution.
    2.  A symmetric positive semi-definite matrix was formed by computing $A A^T$.
    3.  To ensure strict positive definiteness and improve numerical stability, a small multiple of the identity matrix ($10^{-3} I$) was added: $\Sigma_{\text{raw}} = A A^T + 10^{-3} I$.
    4.  The eigenvalues and eigenvectors of $\Sigma_{\text{raw}}$ were computed.
    5.  The final covariance matrix $\Sigma$ was obtained by scaling $\Sigma_{\text{raw}}$ such that its largest eigenvalue became exactly 1.0 ($\Sigma = \Sigma_{\text{raw}} / \lambda_{\max}(\Sigma_{\text{raw}})$). This process results in a full (non-diagonal) covariance matrix with a random orientation and eigenvalues normalized to be within the approximate range (0, 1].
* **Consistency:** The **same** parameter pair $(\mu, \Sigma)$ generated for a specific dimension $d$ was saved and used consistently across all 20 independent training trials conducted for that dimension to ensure comparability between runs.

This defined distribution $N(\mu, \Sigma)$, with its specific center $\mu$ and covariance structure $\Sigma$, served as the basis for sampling.

A training dataset, $\{x_{\text{train}}\}$, was created by drawing a number of random samples directly from this defined distribution $N(\mu, \Sigma)$ (**1,000 samples for $d=2$ and $d=10$; 10,000 samples for $d=100$**). For each training sample $x_{\text{train}}$, the corresponding ground truth Mahalanobis distance, $D_M(x_{\text{train}}) = \sqrt{(x_{\text{train}}-\mu)^T \Sigma^{-1} (x_{\text{train}}-\mu)}$, was computed using the known $\mu$ and $\Sigma$.

To evaluate the trained models, test data was generated dynamically. **Each time an evaluation metric (like the final MSE loss) was computed during analysis, a *new* set of 1,000 independent random samples, $\{x_{\text{test}}\}$, was drawn from the same Gaussian distribution $N(\mu, \Sigma)$.** While this means the specific test set varied between evaluations, it still ensures that models are assessed on unseen data points sharing the identical underlying statistical properties. This approach is deemed sufficient because the experiment's primary goal is to validate the model interpretations and analyze learned parameters, rather than achieving benchmark performance on a fixed test dataset. The exact performance numbers serve mainly to confirm that the models successfully learned the task, enabling subsequent interpretation.

It is worth noting that the Gaussian distribution is characterized by its probability density function falling off exponentially with the squared distance from the mean (specifically, proportional to $e^{-\frac{1}{2} D_M(x)^2}$). Consequently, while samples far from the mean can occur, extreme outliers are statistically rare compared to what might be observed in heavy-tailed distributions. This property makes the synthetic Gaussian data well-suited for this experiment, allowing the focus to remain on the models' capacity to learn the distance metric defined by the distribution's mean and covariance structure, without significant distortion from extreme outlier values.

---

## Experimental Methodology

### Initialization

To ensure a fair comparison within each independent trial while providing a reasonable starting point for learning, a controlled initialization strategy was used for both weights and biases. For each of the 20 trials:

1.  **Weights ($W$):** A single set of initial random weights $W$ was generated using **Kaiming Normal initialization (`torch.nn.init.kaiming_normal_` with `nonlinearity='relu'`)** for the largest linear layer size required (`2d` nodes for `MD_ReLU` and `MD_Sigmoid`).
    * The `MD_ReLU` and `MD_Sigmoid` models were initialized directly with these weights $W$.
    * The `MD_Abs` model, requiring fewer nodes ($d$), was initialized using a subset of these same initial weights (e.g., `W[:d]`).

2.  **Biases ($b$):** The biases $b$ were *not* set to zero. Instead, for each node $i$ (with initial weight vector $w_i$), a single data point $x_{\text{sample}}$ was randomly selected from the training dataset, and the bias was initialized as $b_i = -w_i^T x_{\text{sample}}$.
    * **Rationale:** This initialization ensures that each initial hyperplane $w_i^T x + b_i = 0$ passes through at least one point from the data distribution. This makes it likely that the hyperplane intersects the main data cluster from the start, which is particularly important for preventing "dead neurons" in the `MD_ReLU` model (where the activation might otherwise be zero for all inputs).
    * **Alternative Considerations:** We explicitly avoided initializing with $b=0$, which is often suitable only when input data is pre-centered. We also avoided the theoretically "ideal" initialization $b_i = -w_i^T \mu$ (where $\mu$ is the true mean), as providing the true mean directly via the bias would make the task of learning the cluster center trivial and weaken the demonstration that the network learns this property through optimization. Initializing with respect to a random data sample provides a non-trivial starting point while still encouraging the hyperplanes to be relevant to the data distribution.

This combined approach for weights and biases ensures that differences in performance within a single trial are primarily due to the model architecture and activation function, starting from a shared, reasonably positioned initial state relative to the data.

### Training Process

Each model was trained using the following standard procedure:

* **Optimizer:** Adam (`torch.optim.Adam`)
* **Learning Rate:** $1 \times 10^{-3}$
* **Adam Betas:** $(0.9, 0.99)$
* **Epochs:** 10,000
* **Loss Function:** Mean Squared Error (MSE) loss (`nn.MSELoss`), calculated between the model's predicted distance and the true Mahalanobis distance $D_M(x)$.
* **Device:** Training was performed on a CUDA-enabled GPU if available, otherwise on the CPU.
* **Data Handling:** Training utilized **full-batch gradient descent** (i.e., the entire training dataset was processed in each epoch). Given that the data is synthetically generated from a Gaussian distribution, which inherently lacks heavy tails and extreme outliers, the gradient computed on the full batch provides a stable estimate of the true gradient. This simplifies the optimization dynamics compared to mini-batch or stochastic approaches, aligning with the experiment's focus on analyzing the converged state and learned parameters in a controlled setting.
* **Convergence:** An epoch count was selected to ensure an observed convergence for each dimenion. 2d trained for 10000 epochs; 10d for 50000 epochs; and 100d for 250000 epochs.

### Multiple Trials and Data Collection

The entire training process, starting from generating a new set of initial weights, was repeated **20 times independently** for each model architecture (`MD_Abs`, `MD_ReLU`, `MD_Sigmoid`). After training, the key performance and interpretability metrics (final MSE loss, mean centering distance, sphericity norm, eigenstructure similarity, etc.) are **collected for all 20 resulting models**. This allows for statistical analysis across runs, such as calculating the mean, standard deviation, and distribution of these metrics, providing insight into the consistency and typical behavior of each model architecture. While the single best-performing model (lowest MSE) might be highlighted or visualized, the analysis considers the results from all independent trials.

### Rationale for Analysis Approach

The primary goal of this experiment is not solely to minimize the prediction error, but rather to **analyze the internal representations and parameters learned by models attempting to solve the task**, and to evaluate how well these align with the theoretical interpretations explored in this project.

Neural network training is sensitive to initialization, often leading to different local minima. By running multiple trials and analyzing the results statistically, we aim to understand the typical properties of the solutions found. It should also be noted that the model architectures, particularly `MD_Abs`, are intentionally designed to closely mirror the mathematical structure of the Mahalanobis distance calculation derived in our theory. Therefore, demonstrating that these models *can* learn the distance is expected; the primary contribution lies in analyzing *whether* the learned parameters and internal representations consistently align with the specific interpretations (mean centering, eigenstructure recovery, latent space sphericity, etc.) predicted by the theory across the various runs. The focus is on understanding the characteristics of the solutions found, rather than solely optimizing performance or exhaustively analyzing runs that may have failed to converge effectively.

---

Okay, no problem. Let's reconstruct the "Analysis Methodology" section for the README based on the points you provided and our previous discussion about analyzing all runs.

---

## Analysis Methodology

The experimental results are analyzed to evaluate both overall performance consistency and the alignment of learned parameters with theoretical interpretations across all independent runs.

First, overall performance statistics are collected across all 20 independent training runs for each model architecture (`MD_Abs`, `MD_ReLU`, `MD_Sigmoid`). This includes the mean, variance, minimum, and maximum of the final Mean Squared Error (MSE) loss, providing insight into the typical performance and reliability of each approach.

Second, a detailed analysis is performed by examining the distributions and statistical summaries of key interpretability metrics collected from **all 20 trained models** for each architecture type. While the single best-performing model (lowest MSE) may be used for illustrative visualizations, the core analysis considers the results across all runs:

* **ReLU Mirrored Weight Detection:** For the `MD_ReLU` model runs, the analysis specifically includes detecting pairs of weight vectors where one is approximately the negative of the other ($w_j \approx -w_i$). The frequent observation of such pairs is interpreted as empirical evidence supporting the theory that pairs of ReLU nodes learn to mimic the absolute value function via $Abs(z) = ReLU(z) + ReLU(-z)$. Identifying these pairs allows for **deduplication** – considering only one vector from each mirrored pair – in subsequent analyses (like sphericity and eigenstructure) where the unique learned directions are of primary interest.

* **Mean Centering Verification:** We test the theory that Abs and ReLU activations encourage learned hyperplanes to intersect at cluster centers. This is done by calculating the normalized distance from the true Gaussian mean $\mu$ to the hyperplane defined by each node $(w_i, b_i)$ for every trained model. The distribution (e.g., mean, standard deviation) of these distances across the 20 runs is analyzed. Small average distances support the theory.

* **Latent Space Sphericity Test:** We investigate whether the learned linear transformation $W$ typically "whitens" the data, making the resulting latent space distribution spherical, as suggested by the connection to Mahalanobis distance. For each run, this is assessed using the Frobenius norm $|| (M/\bar{d}) - I ||_F$, where $M = W \Sigma W^T$ (using the unnormalized learned $W$) and $\bar{d}$ is the mean of the diagonal elements of $M$. The distribution of these norms across the 20 runs indicates how consistently the models learn transformations that produce a spherical covariance structure (up to a scaling factor).

* **Eigenstructure Recovery and Comparison:** We test the hypothesis that the learned weights $W$ capture the principal components (eigenstructure) of the input data's covariance matrix $\Sigma$. For each run, eigenvalues ($\lambda_{\text{est}} \approx 1 / ||w_i||^2$) and eigenvector directions ($v_{\text{est}} \propto w_i / ||w_i||$) are estimated from the learned weights. These are compared quantitatively (e.g., using optimal cosine similarity matching) to the true eigenvalues and eigenvectors of $\Sigma$. Statistical summaries of the similarity metrics across the 20 runs assess how reliably the models learn the underlying data structure. The related Mahalanobis matrix approximation ($||W_{\text{norm}}^T W_{\text{norm}} - \Sigma^{-1}||_F$) is also computed across runs to further probe the learned transformation.

* **Visualization:** For the 2D experiments, plots are generated showing the input data distribution, the learned hyperplanes (potentially from the best run or a representative run), and potentially the learned distance contours. These visualizations aid in qualitatively assessing the geometric properties of the learned solutions.

### Rationale for Analysis Approach

The primary goal of this experiment is not solely to minimize the prediction error, but rather to **analyze the internal representations and parameters learned by models attempting to solve the task**, and to evaluate how well these align with the theoretical interpretations presented in the paper.

Neural network training is sensitive to initialization, often leading to different local minima. By running multiple trials and analyzing the results statistically, we aim to understand the typical properties of the solutions found. It should also be noted that the model architectures, particularly `MD_Abs`, are intentionally designed to closely mirror the mathematical structure of the Mahalanobis distance calculation derived in our theory. Therefore, demonstrating that these models *can* learn the distance is expected; the primary contribution lies in analyzing *whether* the learned parameters and internal representations consistently align with the specific interpretations (mean centering, eigenstructure recovery, latent space sphericity, etc.) predicted by the theory across the various runs. The focus is on understanding the characteristics of the solutions found, rather than solely optimizing performance or exhaustively analyzing runs that may have failed to converge effectively.

