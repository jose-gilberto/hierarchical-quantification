import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from .tree import PredictiveClusteringTreeEngine


class HiPACC(BaseEstimator):
  """ Hierarchical Probabilistic Adjusted Classify and Count Tree
  """
  def __init__(self,
               min_samples_leaf: int = 5,
               max_depth: int = 10,
               val_size: float = 0.3,
               random_state: int = 42):
    super().__init__()
    self.min_samples_leaf = min_samples_leaf
    self.max_depth = max_depth
    self.val_size = val_size
    self.random_state = random_state

    self.pct = PredictiveClusteringTreeEngine(min_samples_leaf, max_depth)

    self.tps_ = None
    self.fps_ = None

  def fit(self, X: np.ndarray, y: np.ndarray):
    # 1. internal validation split
    X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=self.val_size, random_state=self.random_state
    )
    
    # 2. fit the custom pct
    self.pct.fit(X_train, y_train)
    
    # 3. calibrate
    # get predictions (prototypes) for validation set
    # since PCT leaves are mean vectors, these are effectively probabilities
    val_preds = self.pct.predict_proba(X_val)

    n_nodes = y.shape[1]
    self.tps_ = np.zeros(n_nodes)
    self.fps_ = np.zeros(n_nodes)

    for j in range(n_nodes):
      pos_mask = y_val[:, j] == 1
      neg_mask = y_val[:, j] == 0

      # TPR estimation: average prediction on positive samples
      if np.any(pos_mask):
        self.tps_[j] = np.mean(val_preds[pos_mask, j])
      else:
        self.tps_[j] = 1.0 # fallback

      # FPR estimation: average prediction on negative samples
      if np.any(neg_mask):
        self.fps_[j] = np.mean(val_preds[neg_mask, j])
      else:
        self.fps_[j] = 0.0 # fallback

    return self
  
  def predict(self, X: np.ndarray) -> np.ndarray:
    # 1. Classify (PCC)
    test_preds = self.pct.predict_proba(X)
    pcc_prev = np.mean(test_preds, axis=0)

    # 2. Adjust (PACC)
    # p_adj = (p_test - fpr) / (tpr - fpr)
    denominator = self.tps_ - self.fps_
    # 2.1. avoid div by 0
    valid_denom = np.abs(denominator) > 1e-4
    pacc_prev = np.zeros(pcc_prev)
    # 2.2. apply correction
    pacc_prev[valid_denom] = (
      (pcc_prev[valid_denom] - self.fps_[valid_denom]) / denominator[valid_denom]
    )
    # Fallback when unstable
    pacc_prev[~valid_denom] = pcc_prev[~valid_denom]

    return np.clip(pacc_prev, 0.0, 1.0)
