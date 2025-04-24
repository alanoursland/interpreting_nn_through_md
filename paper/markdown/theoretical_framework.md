## Theoretical Framework

This section develops the mathematical foundation connecting neural network components to the Mahalanobis distance, providing a framework for interpreting neural network operations through statistical distance metrics.

### 1. Mahalanobis Distance in Context

We consider data points $x \in \mathbb{R}^d$ drawn from a multivariate Gaussian distribution with mean $\mu \in \mathbb{R}^d$ and covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$. The Mahalanobis distance, $D_M(x)$, quantifies the distance from a point $x$ to the mean $\mu$, taking into account the covariance structure of the data. It is formally defined as:

$$D_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}$$

### 2. Interpretation via Principal Component Analysis (PCA) and Whitening

Principal Component Analysis (PCA) provides insight into the structure underlying the Mahalanobis distance. Decomposing the covariance matrix using eigenvalue decomposition yields $\Sigma = V \Lambda V^\top$, where the columns of $V$ are the orthogonal eigenvectors $v_i$ of $\Sigma$, and $\Lambda$ is the diagonal matrix of corresponding eigenvalues $\lambda_i$ representing variance along these principal components. Substituting this into the Mahalanobis distance definition gives:

$$D_M(x) = \sqrt{(x - \mu)^\top V \Lambda^{-1} V^\top (x - \mu)}$$

By applying a change of coordinates corresponding to a projection onto the eigenvectors ($V^\top$) and scaling by the inverse standard deviation ($\Lambda^{-1/2}$), the distance simplifies to the Euclidean ($\ell_2$) norm in this transformed, or "whitened," space:

$$D_M(x) = \left\| \Lambda^{-1/2} V^\top (x - \mu) \right\|_2$$

This demonstrates that the Mahalanobis distance measures Euclidean distance in a space where the data distribution has been sphericized.

### 3. Analysis of a Single Component

Focusing on the contribution to the distance along a single principal component, defined by eigenvector $v_i$ and eigenvalue $\lambda_i$, we can isolate the absolute contribution along this axis:

$$y_i = \left| \lambda_i^{-1/2} v_i^\top (x - \mu) \right|$$

This scalar value, $y_i$, represents the number of standard deviations ($\sqrt{\lambda_i}$) the point $x$ lies from the mean $\mu$ along the principal direction $v_i$. Since $y_i$ measures the deviation from the mean $\mu$ along a principal direction $v_i$ in standard deviation units, its computation is potentially valuable for interpreting learned representations in terms of distance from statistical prototypes. We now examine if this quantity relates to standard neural network operations.

### 4. Equivalence to a Linear Layer with Absolute Value Activation

The algebraic form of $y_i$ can be directly related to neural network components. Let us define a weight vector $W_i = \lambda_i^{-1/2} v_i^\top$ and a corresponding bias term $b_i = -W_i \mu$. Substituting these definitions into the expression for $y_i$ yields:

$$y_i = |W_i x - W_i \mu| = |W_i x + b_i|$$

This resulting expression, $y_i = |W_i x + b_i|$, is formally identical to the computation performed by a standard linear layer (computing the affine transformation $W_i x + b_i$) followed by an absolute value (Abs) activation function. Thus, a linear node with an Abs activation can be interpreted as computing the normalized statistical distance (in standard deviations) of the input $x$ from a prototype $\mu$ along a specific principal axis $v_i$ of the data distribution.

### 5. Realization using ReLU Activations

This connection can be extended to the widely used Rectified Linear Unit (ReLU) activation function. We can construct the absolute value function, and thus the quantity $y_i = |W_i x + b_i|$, using a pair of parallel ReLU neurons. Define a duplicated and negated weight/bias structure:

$$W'_i = \begin{bmatrix} W_i \\ -W_i \end{bmatrix}, \quad b'_i = \begin{bmatrix} b_i \\ -b_i \end{bmatrix}$$

Applying the ReLU activation to an input $x$ using this structure results in a two-dimensional output vector:

$$z'_i = \text{ReLU}(W'_i x + b'_i) = \begin{bmatrix} \text{ReLU}(W_i x + b_i) \\ \text{ReLU}(-W_i x - b_i) \end{bmatrix}$$

Crucially, for any input $x$, at most one of the components of $z'_i$ is non-zero. Consequently, the $\ell_2$ norm (Euclidean norm) of this output vector $z'_i$ simplifies directly to the absolute value:

$$\|z'_i\|_2 = \left\| \text{ReLU}(W'_i x + b'_i) \right\|_2 = \sqrt{\text{ReLU}(W_i x + b_i)^2 + \text{ReLU}(-W_i x - b_i)^2} = |W_i x + b_i|$$

Therefore, we find that:

$$y_i = \|z'_i\|_2 = \|\text{ReLU}(W'_i x + b'_i)\|_2$$

This demonstrates that the statistically meaningful quantity $y_i$—the normalized deviation along a principal axis—can be computed as the Euclidean norm of the output from a simple structure composed of standard ReLU-activated neurons. Mathematically, this result holds for any $\ell_p$ norm ($p \ge 1$) applied to the output vector, but using the $\ell_2$ norm specifically aligns conceptually with the structure of the Mahalanobis distance itself. This finding supports the interpretation of ReLU-based network layers, at least in part, through the lens of statistical distance measures from learned prototypes.
