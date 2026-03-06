import numpy as np
from sklearn.base import clone


class LocalLevelQuantifiers:
  """
  Hierarchical Quantifier based on Local Classifier per Level (LCL).
  Train a quantifier for each level in the hierarchy and applies a top-down correction
  on each estimated prevalence during inference time.
  """

  def __init__(self, base_quantifier, sep="."):
    """
    :param base_quantifier: quantifier instance from quantificationlib (ex: CC)
    :param sep: character that splits each hierarchy level on labels
    """
    self.base_quantifier = base_quantifier
    self.sep = sep
    self.models_ = {}               # Store a quantifier for each one of the levels
    self.classes_per_level_ = {}    # Maps which classes belong to each level
    self.max_level_ = 0
    
  def _map_hierarchy(self, y: np.ndarray) -> dict[int, set[str]]:
    """ Finds the maximum level and the each label per level."""
    self.max_level_ = 0
    self.classes_per_level_ = {}

    for label in y:
      parts = label.split(self.sep)
      depth = len(parts)
      self.max_level_ = max(self.max_level_, depth)

      for i in range(1, depth + 1):
        lvl_label = self.sep.join(parts[:i])
        if i not in self.classes_per_level_:
          self.classes_per_level_[i] = set()
        self.classes_per_level_[i].add(lvl_label)

    # Sort the classes to secure the hierarchical consistency of indexes in matrices
    for lvl in self.classes_per_level_:
      self.classes_per_level_[lvl] = sorted(list(self.classes_per_level_[lvl]))
      
  def fit(self, X: np.ndarray, y: np.ndarray):
    """
    Filter the data and train each quantifier per level.
    """
    self._map_hierarchy(y)

    for level in range(1, self.max_level_ + 1):
      valid_indices = []
      y_level_raw = []

      # Filter only instances that belongs to that level
      for i, label in enumerate(y):
        parts = label.split(self.sep)
        if len(parts) >= level:
          valid_indices.append(i)
          y_level_raw.append(self.sep.join(parts[:level]))
              
      if not valid_indices:
        continue

      classes_lvl = self.classes_per_level_[level]

      if len(classes_lvl) == 1:
        self.models_[level] = "SINGLE_NODE"
        continue
      X_lvl = np.array(X)[valid_indices]

      class_to_idx = {cls: idx for idx, cls in enumerate(classes_lvl)}

      # the quantificationlib requires that y must be an array with integers (0, 1, 2...)
      y_lvl_int = np.array([class_to_idx[lbl] for lbl in y_level_raw])
      
      # Clone the base quantifier to garantee that the model does not cause any problem
      # between levels
      model = clone(self.base_quantifier)
      model.fit(X_lvl, y_lvl_int)

      self.models_[level] = model

    return self
  
  def quantify(self, X_bag: np.ndarray) -> dict[str, float]:
    """
    Estimate the prevalence for a test set (bag) applying a top-down correction.
    Returns a linear dict with all classes and their respective corrected prevalences.
    """
    prevalences_per_level = {}
    
    for level in range(1, self.max_level_ + 1):
      if level not in self.models_:
        continue

      model = self.models_[level]
      classes_lvl = self.classes_per_level_[level]

      if self.models_[level] == "SINGLE_NODE":
        raw_prev = np.array([1.0])
      else:
        model = self.models_[level]
        try:
          raw_prev = model.predict(X_bag)
        except (ValueError, AttributeError):
          raw_prev = model.quantify(X_bag) 

        raw_prev = np.atleast_1d(raw_prev).astype(float)

      adjusted_prev = np.zeros_like(raw_prev, dtype=float)

      if level == 1:
        # The first level dont have a parent, the total prevalence is 
        # the sum of each leaf or children node
        # This consider that we have a Root node
        adjusted_prev = raw_prev
      else:
        # Correction top-down for each other level
        parent_level = level - 1
        classes_parent = self.classes_per_level_[parent_level]
        prev_parent = prevalences_per_level[parent_level]

        # Group prevalences per parent node
        parent_to_children_sum = {}
        for idx, cls_name in enumerate(classes_lvl):
          parent_name = self.sep.join(cls_name.split(self.sep)[:-1])
          if parent_name not in parent_to_children_sum:
            parent_to_children_sum[parent_name] = 0.0
          parent_to_children_sum[parent_name] += raw_prev[idx]

        # Adjust the children to have their sum equals to the father prevalence
        for idx, cls_name in enumerate(classes_lvl):
          parent_name = self.sep.join(cls_name.split(self.sep)[:-1])

          if parent_name in classes_parent:
            idx_parent = classes_parent.index(parent_name)
            p_parent = prev_parent[idx_parent]
            soma_bruta_filhos = parent_to_children_sum[parent_name]

            if soma_bruta_filhos > 0:
              adjusted_prev[idx] = p_parent * (raw_prev[idx] / soma_bruta_filhos)
            else:
              adjusted_prev[idx] = 0.0

      prevalences_per_level[level] = adjusted_prev

    # Squeeze the dict in the format compatible with metrics.evaluators: {'A': 0.5, 'A.B': 0.3}
    final_dict = {}
    for level, prevs in prevalences_per_level.items():
      for cls_name, p in zip(self.classes_per_level_[level], prevs):
        final_dict[cls_name] = p

    return final_dict