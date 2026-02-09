import numpy as np
from typing import Optional, List
from .criterion import find_best_split


class PCTNode:
  def __init__(self, depth: int, prototype: np.ndarray) -> None:
    self.depth = depth
    self.prototype = prototype # the mean vector (probabilities)
    self.feature_idx: Optional[int] = None
    self.threshold: Optional[float] = None
    self.left: Optional["PCTNode"] = None
    self.right: Optional["PCTNode"] = None
    self.is_leaf = True

class PredictiveClusteringTreeEngine:
  """ A custom implementation of a global predictive clustering tree.

  Attributes:
    min_samples_leaf (int): Minimum samples required to split a node
    max_depth (int): maximum depth of the tree
  """
  def __init__(self, min_samples_leaf: int = 5, max_depth: int = 10) -> None:
    self.min_samples_leaf = min_samples_leaf
    self.max_depth = max_depth
    self.root: Optional[PCTNode] = None

  def fit(self, X: np.ndarray, y: np.ndarray):
    """ Builds the PCT recursively

    Args:
        X (np.ndarray): Np array-like containing all training feature vectors
        y (np.ndarray): Np array-like containing all training labels
    """
    self.root = self._build_tree(X, y, depth=0)
    return self
  
  def _build_tree(self, X: np.ndarray, y: np.ndarray, depth) -> PCTNode:
    # 1. Calculate the prototype (mean vector of current cluster)
    prototype = np.mean(y, axis=0)
    node = PCTNode(depth, prototype)

    n_samples = X.shape[0]
    # 2. Check stop criteria
    if (n_samples < 2 * self.min_samples_leaf) or \
       (self.max_depth is not None and depth >= self.max_depth):
      return node
    
    # 2.1. Check if y is pure (variance == 0), due to optimization
    if np.all(y == y[0]):
      return node
    
    # 3. Find the best split
    _, best_feat, best_threshold = find_best_split(X, y, self.min_samples_leaf)

    if best_feat is None:
      return node # no valid split found
    
    # 4. Apply split
    mask_left = X[:, best_feat] <= best_threshold
    X_left, y_left = X[mask_left], y[mask_left]
    X_right, y_right = X[~mask_left], y[~mask_left]

    # 5. Recursion
    node.feature_idx = best_feat
    node.threshold = best_threshold
    node.is_leaf = False
    node.left = self._build_tree(X_left, y_left, depth + 1)
    node.right = self._build_tree(X_right, y_right, depth + 1)

    return node
  
  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """ Returns the prototype (mean vector) of the leaf for each sample.

    Args:
      X (np.ndarray): _description_

    Returns:
      np.ndarray: _description_
    """
    if self.root is None:
      raise RuntimeError('Model not fitted.')
    
    predictions = []
    for i in range(X.shape[0]):
      predictions.append(self._predict_single(X[i], self.root))
    return np.array(predictions)
  
  def _predict_single(self, X: np.ndarray, node: PCTNode) -> np.ndarray:
    if node.is_leaf:
      return node.prototype
    
    if X[node.feature_idx] <= node.threshold:
      return self._predict_single(X, node.left)
    else:
      return self._predict_single(X, node.right)