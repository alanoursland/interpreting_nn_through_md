# Results: Learning Mahalanobis Distance with Neural Networks

This document summarizes the results of the experiment designed to test the theoretical connection between Mahalanobis distance ($D_M(x)$) and specific neural network components (Abs, ReLU activations). It assumes familiarity with the theoretical framework and the detailed experiment description. We analyze the performance and learned representations of models using Absolute Value (`MD_Abs`), ReLU (`MD_ReLU`), and Sigmoid (`MD_Sigmoid`) activations, compared against a non-trained `Ground Truth` model, when trained to predict $D_M(x)$ for synthetic Gaussian data in 2D, 10D, and 100D.

The core theoretical ideas being tested are:
1.  The Mahalanobis distance involves projecting data onto scaled principal components ($v_i, \lambda_i$) and measuring distance in this whitened space.
2.  A linear layer + Abs activation ($|W_i x + b_i|$) can compute the contribution along one such scaled axis ($y_i = |\lambda_i^{-1/2} v_i^\top (x - \mu)|$).
3.  Pairs of ReLU neurons can potentially mimic the Abs function required in (2).

## Performance Analysis

We assessed performance using Mean Squared Error (MSE) loss, analyzing both the average across 20 independent runs and the results from the single best-performing run (lowest MSE during training).

**Table 1: Model Performance (MSE)**

| Model        | Dimension | Mean MSE (Variance) Across Runs | Min MSE (Best Run ID) | Best Run MSE (New Data)* |
| :----------- | :-------- | :---------------------------- | :-------------------- | :----------------------- |
| **ReLU** | 2D        | 0.0284 (1.94e-3)              | 0.000002 (#10)        | 0.000001                 |
|              | 10D       | **0.0279 (1.70e-5)** | 0.021045 (#8)         | 0.036593                 |
|              | 100D      | **0.0077 (3.60e-5)** | 0.002007 (#15)        | 0.305980 (?)             |
| **Abs** | 2D        | 0.1220 (9.94e-3)              | 0.000000 (#18)        | 0.000000                 |
|              | 10D       | 0.0339 (0.00e+0)              | 0.033395 (#19)        | 0.053095                 |
|              | 100D      | 0.0165 (2.00e-6)              | 0.014961 (#12)        | 0.054974                 |
| **Sigmoid** | 2D        | 0.2675 (7.00e-6)              | 0.262260 (#10)        | 0.269654                 |
|              | 10D       | 0.2743 (1.60e-5)              | 0.264517 (#12)        | 0.352433                 |
|              | 100D      | 0.1274 (2.10e-5)              | 0.122150 (#5)         | 0.411511                 |
| **Ground Truth**| 2D        | 0.0000 (0.00)               | 0.000000 (-)          | 0.000000                 |
|              | 10D       | 0.0000 (0.00)               | 0.000000 (-)          | 0.000000                 |
|              | 100D      | 0.0000 (0.00)               | 0.000000 (-)          | 0.000000                 |

\* *Note: Best Run MSE on new data measures generalization of the model selected for lowest training/validation MSE.* The high value for 100D ReLU suggests its selected 'best' model generalized poorly, potentially making the Mean MSE a more reliable indicator of typical performance.*

* **Observations:**
    * `MD_Abs` and `MD_ReLU` consistently learned the task much better than `MD_Sigmoid`.
    * The best runs of `MD_Abs` and `MD_ReLU` achieved near-perfect approximation in 2D, showcasing their potential.
    * Performance consistency (low variance across runs) improved dramatically for `MD_Abs` and `MD_ReLU` in 10D/100D.
    * `MD_ReLU` generally had the lowest *average* MSE across runs in higher dimensions, but `MD_Abs` often showed excellent performance in its best runs and high consistency.

## Interpretability Analysis

We analyzed learned parameters to assess alignment with the theoretical framework, looking at averages across runs and the specific results from the best run.

**Table 2: Interpretability Metrics (Mean (Standard Deviation) Across 20 Runs)**

| Metric                     | Model        | Dimension | 2D Value          | 10D Value         | 100D Value        |
| :------------------------- | :----------- | :-------- | :---------------- | :---------------- | :---------------- |
| **Mirrored Pairs (ReLU)** | ReLU         | 2D        | 0.40 (0.60)       | ---               | ---               |
| *(Count out of d pairs)* |              | 10D       | ---               | 0.00 (0.00)       | ---               |
|                            |              | 100D      | ---               | ---               | 0.00 (0.00)       |
| **Mean Centering Dist.** | ReLU         | 2D        | 0.37 (0.46)       | 0.18 (0.09)       | 0.57 (0.09)       |
|                            | Abs          | 2D        | 1.01 (1.28)       | **0.07 (0.01)** | **0.04 (0.00)** |
|                            | Sigmoid      | 2D        | 3.57 (1.79)       | 5.12 (0.55)       | 5.28 (0.14)       |
|                            | Ground Truth | All       | 0.00 (0.00)       | 0.00 (0.00)       | 0.00 (0.00)       |
| **Sphericity Norm** | ReLU         | 2D        | 1.90 (0.56)       | 5.74 (2.08)       | 23.08 (0.58)      |
| *(Lower is better)* | Abs          | 2D        | **0.85 (0.71)** | **1.06 (0.00)** | **3.64 (0.18)** |
|                            | Sigmoid      | 2D        | 3.41 (0.03)       | 7.85 (0.25)       | 36.86 (0.30)      |
|                            | Ground Truth | 2D        | 9.1e-7 (0.00)     | 5.8e-6 (0.00)     | 2.4e-3 (0.00)     |

**Table 3: Interpretability Metrics (Single Best Run)**

| Metric                     | Model        | Dimension | 2D Value      | 10D Value     | 100D Value    |
| :------------------------- | :----------- | :-------- | :------------ | :------------ | :------------ |
| **Mirrored Pairs (ReLU)** | ReLU         | 2D        | 2 / 2         | ---           | ---           |
| *(Count)* |              | 10D       | ---           | 0 / 10        | ---           |
|                            |              | 100D      | ---           | ---           | 0 / 100       |
| **Mean Centering Dist.** | ReLU         | 2D        | 0.0012        | 0.178         | 0.781         |
| *(Avg. over nodes)* | Abs          | 2D        | 0.0001        | **0.074** | **0.038** |
|                            | Sigmoid      | 2D        | 3.95          | 6.40          | 5.26          |
|                            | Ground Truth | All       | 0.00          | 0.00          | 0.00          |
| **Sphericity Norm** | ReLU         | 2D        | 0.0066        | 4.97          | 23.69         |
| *(Lower is better)* | Abs          | 2D        | **2.2e-5** | **1.06** | **3.47** |
|                            | Sigmoid      | 2D        | 3.35          | 7.90          | 37.16         |
|                            | Ground Truth | All       | 9.1e-7        | 5.8e-6        | 2.4e-3        |
| **Eigenvalue Recovery** | ReLU         | All       | Poor          | Poor          | Poor          |
| *(Qualitative)* | Abs          | All       | Poor          | Poor          | Poor          |
| **Eigenvector Recovery** | ReLU         | 2D        | Good (w/ pairs) | Poor          | Poor          |
| *(Qualitative)* | Abs          | 2D        | Poor          | Fair* | Fair* |

\* *Based on achieving good sphericity, suggesting alignment with the overall whitening transformation.*

* **Key Observations:**
    * **ReLU Pairing:** The best 2D ReLU model perfectly implemented the pairing mechanism (2 out of 2 pairs mirrored). This mechanism was entirely absent in the best 10D and 100D ReLU models, consistent with the average results.
    * **Mean Centering:** The best `MD_Abs` runs consistently achieved excellent mean centering, especially in 10D/100D. The best `MD_ReLU` runs also showed good centering in 2D, and while worse than `MD_Abs` in higher dimensions, they were still significantly better than `MD_Sigmoid`, potentially offering partial support for the centering aspect of the theory.
    * **Whitening (Sphericity):** The best `MD_Abs` runs achieved excellent sphericity, particularly in 2D, indicating successful learning of the whitening transformation central to the Mahalanobis distance. While the sphericity norm increased in 100D, it remained far better than `MD_ReLU` or `MD_Sigmoid`. The `MD_ReLU` model only achieved good sphericity in its best 2D run where pairing occurred.
    * **Sphericity Caveat (ReLU/Sigmoid):** The sphericity calculation for `MD_ReLU` and `MD_Sigmoid` uses all `2d` outputs. In cases where the pairing mechanism is absent (10D/100D ReLU) or irrelevant (Sigmoid), this metric may not directly correspond to the intended theoretical interpretation of whitening based on `d` principal components. The poor scores reflect a failure to produce a scaled identity covariance in the latent space, but the exact interpretation is less clear than for `MD_Abs`.
    * **Eigenstructure:** Accurate recovery of eigenvalues was poor across all trained models. However, the best `MD_Abs` runs (due to good sphericity) and the best 2D `MD_ReLU` run (due to pairing) likely recovered the principal *directions* (eigenvectors) better than other models/conditions.
    * **Ground Truth:** The analytical `Ground Truth` model provided a perfect benchmark in 2D/10D but showed slightly degraded sphericity in 100D (norm ~2.4e-3 vs ~1e-6), indicating numerical precision limits.

## Discussion: Alignment with Theory

Comparing these results to the theoretical framework yields several insights:

1.  **Core Task Feasibility:** The success of `MD_Abs` and `MD_ReLU` confirms that architectures based on these activations, structured to compute a squared L2 norm in a latent space, *can* effectively approximate the Mahalanobis distance, unlike the `MD_Sigmoid` baseline.

2.  **`MD_Abs` as Theoretical Match:** The `MD_Abs` model aligns well with the theory's geometric interpretation. Its ability (especially in the best runs and consistently in high-D) to learn centered representations and whitening transformations strongly supports the idea that `Linear -> Abs` layers can model the scaled projections onto principal axes ($y_i = |W_i x + b_i|$) required by the Mahalanobis framework. The primary deviation from theory was the lack of precise eigenvalue recovery (scaling).

3.  **`MD_ReLU` Mechanism is Dimension-Dependent:** The theory suggests ReLU achieves this via pairing. Our results show this mechanism is strongly supported *only in 2D*. Its absence in 10D/100D means the `MD_ReLU` model, while achieving low MSE, does *not* align with the specific proposed Mahalanobis interpretation in higher dimensions. It likely relies on other function approximation capabilities of ReLU networks.

4.  **Mean Centering:** Both Abs and ReLU seem predisposed to finding centered solutions compared to Sigmoid. While Abs was clearly superior and more consistent (especially in high-D), ReLU's centering performance (better than Sigmoid) might still be considered partially supportive of the idea that these activation types encourage learning representations relative to data clusters.

5.  **Whitening:** Learning the whitening transformation is key to the Mahalanobis interpretation. `MD_Abs` consistently demonstrated this capability (good sphericity), reinforcing its theoretical alignment. `MD_ReLU` only showed this when the pairing mechanism was active (2D best run).

6.  **Best vs. Average:** The best-run results demonstrate the *potential* of these architectures to align with the theory (e.g., near-perfect `MD_Abs` sphericity in 2D, perfect ReLU pairing in 2D). The average results show how consistently this potential is realized across typical training runs. The divergence between best/average and across dimensions highlights the complexity of optimization and representation learning.

## Summary Conclusion

* The `MD_Abs` architecture provides strong, consistent empirical support for the theoretical interpretation linking linear layers + Abs activation to the geometric components (centering, whitening) of the Mahalanobis distance, particularly in 10D and 100D.
* The interpretation of `MD_ReLU` via neuron pairing mimicking Abs is supported only in the low-dimensional (2D) case. In higher dimensions, `MD_ReLU` fails to align with this mechanism or the associated geometric properties, despite potentially achieving low prediction error.
* Minimizing MSE does not guarantee convergence to a theoretically interpretable solution, as seen in the high-D `MD_ReLU` results.
* The study validates key aspects of the Mahalanobis interpretation framework but underscores the crucial role of dimensionality and architectural choices (Abs vs. ReLU) in how well learned representations align with specific theoretical mechanisms.
* Practical numerical limitations can affect even analytical solutions in high dimensions.