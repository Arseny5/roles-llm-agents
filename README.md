# KL Divergence between Multivariate Gaussians of LLM roles

[Persona-hub](https://arxiv.org/pdf/2406.20094v1) create diverse synthetic data by thematic roles, in one distribution (like hospital -> nurse -> patient). Therefore, I use clustering and dimensionality reduction (HDBSCAN and UMAP) to get a list of responses for a role in order to approximate the distribution of each role. Embeddings of texts are obtained using 2 approaches: BERT-based or TFiDF models.

**TFiDF approach**

```math
D_{KL}(P \parallel Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)
```

where:
- $`P`$ and $`Q`$ are the probability distributions,
- $`P(i)`$ and $`Q(i)`$ are the probability mass functions for the discrete case,

**BERT approach**

The KL divergence between two multivariate Gaussian distributions $`\mathcal{N}(\mu_1, \Sigma_1)`$ and $`\mathcal{N}(\mu_2, \Sigma_2)`$ is given by the following formula:

```math
D_{KL}(\mathcal{N}(\mu_1, \Sigma_1) \parallel \mathcal{N}(\mu_2, \Sigma_2)) =
\frac{1}{2} \left( \text{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{-1} (\mu_2 - \mu_1) - k + \log \left(\frac{\det \Sigma_2}{\det \Sigma_1}\right) \right)
```

where:
- $`\mu_1`$ and $`\mu_2`$ are the means of the distributions
- $`\Sigma_1`$ and $`\Sigma_2`$ are the covariance matrices
- $`k`$ is the dimensionality of the distributions



![llm-condition](personahub-pipe.png)
