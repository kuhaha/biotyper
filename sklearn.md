CWT 
- *Continuous Wavelet Transform*. A CWT performs a convolution with data using the wavelet function, which is characterized by a *width* parameter and *length* parameter. The wavelet function is allowed to be complex.

.. Wavelet function takes 2 arguments. The first argument is the number of points that the returned vector will have (*len(wavelet(length,width)) == length*). The second is a *width* parameter, defining the size of the wavelet (e.g. standard deviation of a gaussian). 

```python
scipy.signal.cwt(
  data, 
  wavelet, 
  widths, 
  dtype=None, 
  **kwargs
)
```

DBSCAN 
- *Density-Based Spatial Clustering of Applications with Noise*. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.
```python
sklearn.cluster.DBSCAN(
	eps=0.5,  # epsilon: maximum distance to be considered as in the neighborhood 
	*, 
	min_samples=5, 
	metric='euclidean', 
	metric_params=None, 
	algorithm='auto', 
	leaf_size=30, 
	p=None, 
	n_jobs=None
)
```

HDBSCAN 
- *Hierarchical Density-Based Spatial Clustering of Applications with Noise*. Performs DBSCAN over varying epsilon values and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN), and be more robust to parameter selection. 

```python
sklearn.cluster.HDBSCAN(
  min_cluster_size=5, 
  min_samples=None, 
  cluster_selection_epsilon=0.0, 
  max_cluster_size=None, 
  metric='euclidean', 
  metric_params=None, 
  alpha=1.0, 
  algorithm='auto', 
  leaf_size=40, 
  n_jobs=None, 
  cluster_selection_method='eom', 
  allow_single_cluster=False, 
  store_centers=None, 
  copy=False
)
```

Kernel Density Estimation 
  - Density estimation walks the line between unsupervised learning, feature engineering, and data modeling. Some of the most popular and useful density estimation techniques are mixture models such as Gaussian Mixtures, and neighbor-based approaches such as the kernel density estimate.
  - Kernel density estimation in scikit-learn is implemented in the KernelDensity estimator, which uses the Ball Tree or KD Tree for efficient queries.   
  - Mathematically, a kernel is a positive function `K(x;h)` which is controlled by the bandwidth parameter. Given this kernel form, the density estimate at a point  within a group of points $x_i; i=1,\cdots,N$ is given by:
   \[
   	\rho_{K}(y)=\sum K(y-x_i;h) 
   \]
  - The bandwidth here acts as a smoothing parameter, controlling the tradeoff between bias and variance in the result. A large bandwidth leads to a very smooth (i.e. high-bias) density distribution. A small bandwidth leads to an unsmooth (i.e. high-variance) density distribution. One can either set manually this parameter or use Scott’s and Silvermann’s estimation methods.

```python
sklearn.neighbors.KernelDensity(
  *, 
  bandwidth=1.0, 
  algorithm='auto', 
  kernel='gaussian', 
  metric='euclidean', 
  atol=0, rtol=0, # absolute/relative tolerance
  breadth_first=True, 
  leaf_size=40, 
  metric_params=None
)
```