# math
- `round()`
```python
f = 123.456
print(round(f))
print(round(f,1))
print(round(f,2))
print(round(f,-1))
print(round(f,-2))
# 123
# 123.5
# 123.46
# 120.0
# 100.0
```

- `quitize()`
```python
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
f = 123.456
print(Decimal(f))
print(Decimal(str(f)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
print(Decimal(str(f)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
print(Decimal(str(f)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
# 123.4560000000000030695446184836328029632568359375
# 123
# 123.5
# 123.46
```
# scikit learn

- Feature Extraction, Feature Selection
  -  Feature extraction is very different from Feature selection: the former consists in transforming arbitrary data, such as text or images, into numerical features usable for machine learning. The latter is a machine learning technique applied on these features
- Preprocessing and Normalization
  - Utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.

## Clustering
Clustering of unlabeled data can be performed with the module **sklearn.cluster**.

Each clustering algorithm comes in two variants: a class, that implements the fit method to learn the clusters on train data, and a function, that, given train data, returns an array of integer labels corresponding to the different clusters. For the class, the labels over the training data can be found in the labels_ attribute.

- `cluster.AffinityPropagation(*[, damping, ...])`, Perform Affinity Propagation Clustering of data.
- `cluster.AgglomerativeClustering([...])`, Agglomerative Clustering.
- `cluster.Birch(*[, threshold, ...])`, Implements the BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) clustering algorithm.
- `cluster.DBSCAN([eps, min_samples, metric, ...])`, Perform DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering from vector array or distance matrix.
- `cluster.HDBSCAN([min_cluster_size, ...])`, Cluster data using hierarchical density-based clustering.
- `cluster.FeatureAgglomeration([n_clusters, ...])`, Agglomerate features.

- `cluster.KMeans([n_clusters, init, n_init, ...])`, K-Means clustering.

- `cluster.BisectingKMeans([n_clusters, init, ...])`, Bisecting K-Means clustering.

- `cluster.MiniBatchKMeans([n_clusters, init, ...])`, Mini-Batch K-Means clustering.

- `cluster.MeanShift(*[, bandwidth, seeds, ...])`, Mean shift clustering using a flat kernel.

- `cluster.OPTICS(*[, min_samples, max_eps, ...])`, Estimate clustering structure from vector array.

- `cluster.SpectralClustering([n_clusters, ...])`, Apply clustering to a projection of the normalized Laplacian.

- `cluster.SpectralBiclustering([n_clusters, ...])`, Spectral biclustering (Kluger, 2003).

- `cluster.SpectralCoclustering([n_clusters, ...])`, Spectral Co-Clustering algorithm (Dhillon, 2001).

## Nearest Neihbors
**sklearn.neighbors** provides functionality for unsupervised and supervised neighbors-based learning methods. Unsupervised nearest neighbors is the foundation of many other learning methods, notably manifold learning and spectral clustering. Supervised neighbors-based learning comes in two flavors: classification for data with discrete labels, and regression for data with continuous labels.

The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

- `neighbors.BallTree(X[, leaf_size, metric])`, BallTree for fast generalized N-point problems
- `neighbors.KDTree(X[, leaf_size, metric])`, KDTree for fast generalized N-point problems
- `neighbors.KernelDensity(*[, bandwidth, ...])`, Kernel Density Estimation.
- `neighbors.KNeighborsClassifier([...])`, Classifier implementing the k-nearest neighbors vote.
- `neighbors.KNeighborsRegressor([n_neighbors, ...])`, Regression based on k-nearest neighbors.
- `neighbors.KNeighborsTransformer(*[, mode, ...])`, Transform X into a (weighted) graph of k nearest neighbors.
- `neighbors.LocalOutlierFactor([n_neighbors, ...])`, Unsupervised Outlier Detection using the Local Outlier Factor (LOF).
- `neighbors.RadiusNeighborsClassifier([...])`, Classifier implementing a vote among neighbors within a given radius.
- `neighbors.RadiusNeighborsRegressor([radius, ...])`, Regression based on neighbors within a fixed radius.
- `neighbors.RadiusNeighborsTransformer(*[, ...])`, Transform X into a (weighted) graph of neighbors nearer than a radius.
- `neighbors.NearestCentroid([metric, ...])`, Nearest centroid classifier.
- `neighbors.NearestNeighbors(*[, n_neighbors, ...])`, Unsupervised learner for implementing neighbor searches.
- `neighbors.NeighborhoodComponentsAnalysis([...])`, Neighborhood Components Analysis.
- `neighbors.kneighbors_graph(X, n_neighbors, *)`, Compute the (weighted) graph of k-Neighbors for points in X.
- `neighbors.radius_neighbors_graph(X, radius, *)`, Compute the (weighted) graph of Neighbors for points in X.
- `neighbors.sort_graph_by_row_values(graph[, ...])`, Sort a sparse graph such that each row is stored with increasing values.


# Signal processing (scipy.signal)
Convolution, B-splines, Filtering, Filter design
## Wavelets
- `cascade(hk[, J])`, Return (x, phi, psi) at dyadic points K/2**J from filter coefficients.
- `daub(p)`, The coefficients for the FIR low-pass filter producing Daubechies wavelets.
- `morlet(M[, w, s, complete])`, Complex Morlet wavelet.
- `qmf(hk)`, Return high-pass qmf filter from low-pass
- `ricker(points, a)`, Return a Ricker wavelet, also known as the "Mexican hat wavelet".
- `morlet2(M, s[, w])`, Complex Morlet wavelet, designed to work with cwt.
- `cwt(data, wavelet, widths[, dtype])`, Continuous wavelet transform.
## Peak Finding 
- `argrelmin(data[, axis, order, mode])`, Calculate the relative minima of data.
- `argrelmax(data[, axis, order, mode])`, Calculate the relative maxima of data.
- `argrelextrema(data, comparator[, axis, ...])`, Calculate the relative extrema of data.
- `find_peaks(x[, height, threshold, distance, ...])`, Find peaks inside a signal based on peak properties.
- `find_peaks_cwt(vector, widths[, wavelet, ...])`, Find peaks in a 1-D array with wavelet transformation.
- `peak_prominences(x, peaks[, wlen])`, Calculate the prominence of each peak in a signal.
- `peak_widths(x, peaks[, rel_height, ...])`, 
Calculate the width of each peak in a signal.
# Spatial algorithms and data structures (scipy.spatial)
- Spatial transformations
- Nearest-neighbor queries
  - *KDTree, cKDTree, Rectangle*, ...
- Distance metrics (scipy.spatial.distance)
  - Distance functions between two numeric vectors `u` and `v`.
    -  *cosine, euclidean, jensenshannon, mahalanobis, minkowski, cityblock, chebyshev*, ...
  - Distance functions between two boolean vectors  (representing sets) `u` and `v`. 
    - *dice, jaccard, hamming, russellrao, sokalmichener, yule*, . .. 

# Statistical functions (scipy.stats)
A large number of probability distributions, summary and frequency statistics, correlation functions and statistical tests, masked statistics, kernel density estimation, quasi-Monte Carlo functionality, and more.

Statistics is a very large area, and there are topics that are out of scope for SciPy and are covered by other packages. Some of the most important ones are:

- **statsmodels**: regression, linear models, time series analysis, extensions to topics also covered by scipy.stats.
- **Pandas**: tabular data, time series functionality, interfaces to other statistical languages.
- **PyMC**: Bayesian statistical modeling, probabilistic machine learning.
- **scikit-learn**: classification, regression, model selection.
- **Seaborn**: statistical data visualization.
- **rpy2**: Python to R bridge.
## Probability distributions
- Continuous distributions, e.g.,
  - **alpha**, An alpha continuous random variable.
  - **anglit**, An anglit continuous random variable.
  - **arcsine**, An arcsine continuous random variable.
- Multivariate distributions, e.g.,
  - **multivariate_normal**, A multivariate normal random variable.
  - **matrix_normal**, A matrix normal random variable.
- Discrete distributions, e.g., 
  - **bernoulli**, A Bernoulli discrete random variable.
  - **betabinom**, A beta-binomial discrete random variable.
  - **binom**, A binomial discrete random variable.