# Neural_Manifolds
A repository of Neural Manifold Learning (NML) techniques described and compared in the review "Neural manifold analysis of brain circuit dynamics in health and disease"

https://link.springer.com/article/10.1007/s10827-022-00839-3

Mitchell-Heggs, R., Prado, S., Gava, G.P., Go, M.A., Schultz, S.R., 2023. Neural manifold analysis of brain circuit dynamics in health and disease. J Comput Neurosci 51, 1–21. https://doi.org/10.1007/s10827-022-00839-3

Recent developments in experimental neuroscience make it possible to simultaneously record the activity of thousands of neurons. However, the development of analysis approaches for such large-scale neural recordings have been slower than those applicable to single-cell experiments. One approach that has gained recent popularity is neural manifold learning. This approach takes advantage of the fact that often, even though neural datasets may be very high dimensional, the dynamics of neural activity tends to traverse a much lower-dimensional space. The topological structures formed by these low-dimensional neural subspaces are referred to as neural manifolds, and may potentially provide insight linking neural circuit dynamics with cognitive function and behavioural performance. In this paper we review a number of linear and non-linear approaches to neural manifold learning, by setting them within a common mathematical framework, and comparing their advantages and disadvantages with respect to their use for neural data analysis. We apply them to a number of datasets from published literature, comparing the manifolds that result from their application to hippocampal place cells, motor cortical neurons during a reaching task, and prefrontal cortical neurons during a multi-behaviour task. We find that in many circumstances linear algorithms produce similar results to non-linear methods, although in particular in cases where the behavioural complexity is greater, nonlinear methods tend to find lower dimensional manifolds, at the possible expense of interpretability. We demonstrate that these methods are applicable to the study of neurological disorders through simulation of a mouse model of Alzheimers Disease, and speculate that neural manifold analysis may help us to understand the circuit-level consequences of molecular and cellular neuropathology.


<img width="1374" alt="image" src="https://user-images.githubusercontent.com/38789733/161796468-ca38b653-ed4a-43e7-b8f4-bc25a9548539.png">

Fig. 1 Schematic showing a typical example of how a manifold learning algorithm may reduce the dimensionality of a high dimensional neural population time series to produce a more interpretable low dimensional representation. A high dimensional neural population activity matrix, X, with N neurons and T time points, is projected into a lower dimensional manifold space and the trajectory visualised in the space formed by the first 3 dimensions, c1, c2 and c3.

#### NML Algorithms:
1. Principle Component Analysis (PCA)
2. Multi-Dimenisonal Scaling (MDS)
3. Isomap
4. Locally Linear Embedding (LLE)
5. Laplacian Eigenmaps (LEM)
6. t-distributed Stochastic Neighbour Embedding (t-SNE)
7. Uniform Manifold Approximation and Projection (UMAP)

#### Manifold analysis Algorithms:
1. Optimal linear estimator (OLE) decoder
2. Reconstruction score (Adapted from https://github.com/jakevdp/pyLLE/blob/master/pyLLE/python_only/LLE.py)
3. Intrinsic dimensionality

#### Data sets used for NML comparison*:
1. Two-photon calcium imaging of hippocampal subfield CA1 in a mouse running along a circular track taken from (Go et al, 2021)
2. Multi-electrode array extracellular electrophysiological recordings from the motor cortex of a macaque performing a radial arm goal-directed reach task from (Yu et al, 2007)
3. Single-photon "mini-scope" calcium imaging data recorded from the prefrontal cortex of a mouse under conditions where multiple task-relevant behavioural variables were monitored from (Rubin et al, 2019)

<img width="952" alt="image" src="https://user-images.githubusercontent.com/38789733/161800388-4ba698c7-a2c1-43b9-a524-e653ff5014b2.png">

*Due to data permissions, individual datasets will not be provded in this repository.  To get access to any data, please reach out to the the corresponding author.  To illustrate how each algorithm can be applied to a dataset we have used the first dataset.

#### Getting started:
1. Install the NML repository
2. Open the jupyter notebook: Neural_Manifold_Learning_CA1.ipynb
3. Adapt code to your own data N x T (Neural traces/events/spikes over time) and a behavioural vector
