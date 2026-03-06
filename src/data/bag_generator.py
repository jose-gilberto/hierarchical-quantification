import numpy as np


def generate_hierarchical_bags(X: np.ndarray,
                               y: np.ndarray,
                               bag_size: int = 100,
                               n_bags_uniform: int = 5,
                               n_bags_extreme: int = 5) -> dict[str, any]:
  """
  Generate each test bag by mixing uniform and extreme shifts.
  
  In the case that a specific class doesnt have suficient instances to supply the sorted
  prevalence, each other instances belonging to that class will be used (without reposition),
  resulting in a smaller bag.
  """
  X_array, y_array = np.array(X), np.array(y)

  # Identify each leaf
  leaf_classes = np.unique(y_array)
  n_leaves = len(leaf_classes)

  # Maps each indice in X that belongs the each leaf
  indices_per_leaf = {cls_: np.where(y_array == cls_)[0] for cls_ in leaf_classes}
  
  bags = []
  scenarios = [(n_bags_uniform, 1.0), (n_bags_extreme, 0.1)]

  for n_bags, alpha in scenarios:
    for _ in range(n_bags):
      # Sort the target prevalence of leafs using Dirichlet
      target_prevs = np.random.dirichlet(np.ones(n_leaves) * alpha)

      # Converts the prevalence to the ideal absolute size
      counts = np.round(target_prevs * bag_size).astype(int)

      # Initial fine tuning before cut off size due to few leaf instances
      diff = bag_size - counts.sum()
      if diff != 0:
        counts[np.argmax(counts)] += diff

      bag_indices = []
      
      # Collect each data to secure that dont surpass the limit
      for i, cls in enumerate(leaf_classes):
        available_idx = indices_per_leaf[cls]
        n_available = len(available_idx)
        
        # We take the target or the maximum available
        n_to_sample = min(counts[i], n_available)

        if n_to_sample > 0:
          # replace=False secure that none instances is duplicated
          chosen_idx = np.random.choice(available_idx, size=n_to_sample, replace=False)
          bag_indices.extend(chosen_idx)

      np.random.shuffle(bag_indices)

      # Only store a bag if there is at least 1 valid element on that bag
      if len(bag_indices) > 1:
        bags.append({
          'X': X_array[bag_indices],
          'y': y_array[bag_indices], 
          'alpha_shift': alpha,
          'target_size': bag_size,
          'actual_size': len(bag_indices)
        })

  return bags