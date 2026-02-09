"""
In PCTs, the prototype of a cluster (leaf) is the mean vector $\bar{y}$. The
quality of a cluster is measured by its Sum of Squared Errors (SSE).

$$
SSE(S) = \sum_{y \in S} \sum_{j=1}^{\text{targets}} (y_j - \bar{y}_j)^2
$$

To find the best split, we maximize the Variance Reduction (which is 
computationally equivalent to maximizing the sum of the explained squared sums
of the children).
"""
import numpy as np
from typing import Tuple, Optional


def find_best_split(X: np.ndarray, y: np.ndarray, min_samples_leaf: int) -> Tuple[float, int, float]:
  """
  Finds the best split for a predictive clustering tree (PCT).
  
  Optimized using numpy vectorization to maximize the reduction in
  multi-target variance (Global Approach).
  
  Returns:
    (best_score, best_feature_idx, best_threshold)
  """
  n_samples, n_features = X.shape
  # Current impurity (not strictly neeeded for comparison, but conceptual)
  # we want to maximize (Sum_left^2 / N_left) + (Sum_right^2 / N_right)
  # this is equivalent to minimize weighted variance.
  
  best_score = -np.inf
  best_feat = None
  best_thresh = None
  
  # pre-calculate global sum for constant time updates
  G = y.sum(axis=0)
  
  for feat_idx in range(n_features):
    # sort data by current feature
    sorted_indices = np.argsort(X[:, feat_idx])
    X_sorted = X[sorted_indices, feat_idx]
    y_sorted = y[sorted_indices]
    
    # cumulative sums of the target vectors Y to calculate
    # means quickly 
    # (n_samples, n_target)
    y_cumsum = np.cumsum(y_sorted, axis=0)
    
    # identify valid split positions (where feature value changes)
    # only check unique values to avoid splitting strictly identical features
    unique_values, unique_indices = np.unique(X_sorted, return_index=True)
    
    # skip the first index (0 items on left) and iterate valid split points
    # unique_indices points to the first occurrence
    # split after these
    # we start from 1 because we need at least min_samples_leaf
    for i in range(1, len(unique_indices)):
      split_idx = unique_indices[i]
      # enforce min_samples_leaf constraint
      if split_idx < min_samples_leaf or (n_samples - split_idx) < min_samples_leaf:
        continue
      
      # vectorized variance reduction calculation
      # left child stats
      n_l = split_idx
      sum_l = y_cumsum[split_idx - 1]
      
      # right child stats
      n_r = n_samples - n_l
      sum_r = G - sum_l
      
      # the score to maximize is the sum of squared means weighted by n
      # score = n_l * ||mean_l||^2 + n_r * ||mean_r||^2
      #       = n_l * ||sum_l / n_l||^2 ...
      #       = (sum_l^2 / n_l) + (sum_r^2 / n_r)
      # we sum over all target dims (axis=0 of the squared vectors)
      score = np.sum((sum_l ** 2) / n_l) + np.sum((sum_r ** 2) / n_r)
      
      if score > best_score:
        best_score = score
        best_feat = feat_idx
        # threshold is midpoint between current value and previous value
        best_thresh = (X_sorted[split_idx] + X_sorted[split_idx - 1]) / 2.0

  return best_score, best_feat, best_thresh