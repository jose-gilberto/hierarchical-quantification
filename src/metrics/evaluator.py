import numpy as np
from collections import Counter


def base_weight_method(level: int) -> float:
  return 1.0


class HierarchicalPrevalenceEvaluator:
  
  def __init__(self, sep: str = '.'):
    self.sep = sep
  
  def calculate_true_prevalences(self, y_true: np.ndarray) -> dict[int, dict[str, float]]:
    """ Calculate the true prevalence per level
    """
    n_samples = len(y_true)
    level_counts = {}

    # label -> A.B.C
    for label in y_true:
      parts = label.split(self.sep)
      # 1, ..., max_level
      for i in range(1, len(parts) + 1):
        # A, A.B, A.B.C
        lvl_label = self.sep.join(parts[:i])
        if i not in level_counts:
          level_counts[i] = []
        level_counts[i].append(lvl_label)
        
    level_prevalences = {}
    for level, items in level_counts.items():
      counts = Counter(items)
      level_prevalences[level] = {
        cls_: count / n_samples for cls_, count in counts.items()
      }
    
    # OBS: the count is assumed to be related to the total number of samples
    # each node in the hierarchy receive a prevalence related to the global count
    return level_prevalences
  
  def evaluate_from_prevalences(self,
                                y_true: np.ndarray,
                                pred_prevalences_dict: dict[str, float],
                                weigth_method: callable = base_weight_method) -> tuple[dict[int, float], float]:
    """ Compares the y_true with a dictionary of predicted prevalences.
    the dictionary must follow the format {'A': 0.5, 'A.B': 0.3, ...}
    """
    prev_true = self.calculate_true_prevalences(y_true)

    prev_pred = {}
    for cls_, p in pred_prevalences_dict.items():
      level = len(cls_.split(self.sep))
      if level not in prev_pred:
        prev_pred[level] = {}
      prev_pred[level][cls_] = p
      
    max_level = max(max(prev_true.keys(), default=0), max(prev_pred.keys(), default=0))
    
    errors_per_level = {}
    total_weighted_error = 0.0
    total_weight = 0.0
    
    for level in range(1, max_level + 1):
      true_lvl = prev_true.get(level, {})
      pred_lvl = prev_pred.get(level, {})
      
      all_classes_in_level = set(true_lvl.keys()).union(set(pred_lvl.keys()))
      if not all_classes_in_level:
        continue
      
      level_abs_errors = []
      for cls_ in all_classes_in_level:
        p_true = true_lvl.get(cls_, 0.0)
        p_pred = pred_lvl.get(cls_, 0.0)
        level_abs_errors.append(abs(p_pred - p_true))
        
      sum_ae_level = np.sum(level_abs_errors)
      errors_per_level[level] = sum_ae_level
      
      w = weigth_method(level)
      total_weighted_error += w * sum_ae_level
      total_weight += w
  
    h_ae = total_weighted_error / total_weight if total_weight > 0 else 0.0
    return errors_per_level, h_ae
