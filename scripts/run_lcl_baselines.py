import yaml
import json
import os
import pandas as pd
import numpy as np
import argparse
import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from quantificationlib.baselines.cc import CC, PCC
from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.multiclass.df import HDX, HDy , DFX, DFy, MMy
from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.energy import EDX, EDy

import sys
sys.path.append('../')

from src.data.bag_generator import generate_hierarchical_bags
from src.metrics.evaluator import HierarchicalPrevalenceEvaluator
from src.quantifiers.level import LocalLevelQuantifiers

CLASSIFIERS_FACTORY = {
  "rf": lambda seed: RandomForestClassifier(random_state=seed),
  "lr": lambda seed: LogisticRegression(random_state=seed, max_iter=1000),
  "svm": lambda seed: SVC(probability=True, random_state=seed)
}

QUANTIFIERS_FACTORY = {
  "cc": CC,
  "pcc": PCC,
  "ac": AC,
  "pac": PAC,
  # "hdx": HDX,
  "hdy": HDy,
  # "dfx": DFX,
  "dfy": DFy,
  # "mmy": MMy,
  "em": EM,
  # "edx": EDX,
  "edy": EDy,
}


def load_config(config_path: str) -> dict[str, any]:
  with open(config_path, 'r') as f:
    return yaml.safe_load(f)


def extract_embeddings(df):
  X = [eval(e) if isinstance(e, str) else e for e in df['embeddings'].values]
  return np.stack(X) if len(np.array(X).shape) == 1 else np.array(X)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default="baselines_config.yaml", help="path to the yaml config file")
  args = parser.parse_args()

  config = load_config(args.config)
  seed = config.get('random_seed', 42)

  np.random.seed(seed)

  print(f"Reading the dataset...")
  train = pd.read_parquet(config['train_path'])
  X_train, y_train = extract_embeddings(train), train['label'].values
  
  test = pd.read_parquet(config['test_path'])
  X_test, y_test = extract_embeddings(test), test['label'].values

  print('Generating test bags...')
  bags = generate_hierarchical_bags(X_test,
                                    y_test,
                                    bag_size=config['bag_size'],
                                    n_bags_uniform=config['n_bags_uniform'],
                                    n_bags_extreme=config['n_bags_extreme'])
  
  evaluator = HierarchicalPrevalenceEvaluator(sep='.')

  results_path = config['output_path']
  if not os.path.exists(results_path):
    pd.DataFrame(columns=[
      'bag_idx', 'true_prevalences', 'pred_prevalences', 'hae', 'quantifier', 'classifier', 'strategy'
    ]).to_csv(results_path, index=False)

  print('Starting the experiments...')

  for clf_name in config['classifiers']:
    for quant_name in config['quantifiers']:
      print(f"-> Training: Strategy={config['strategy']} | Quantifier={quant_name} | Classifier={clf_name}")

      base_clf = CLASSIFIERS_FACTORY[clf_name](seed)
      if quant_name in ['cc', 'pcc']:
        base_quant = QUANTIFIERS_FACTORY[quant_name](base_clf)
      else:
        base_quant = QUANTIFIERS_FACTORY[quant_name](estimator_train=base_clf, estimator_test=base_clf)
      

      model = LocalLevelQuantifiers(base_quantifier=base_quant, sep=".")
      model.fit(X_train, y_train)
      
      resultados_batch = []
      for bag_idx, bag in tqdm.tqdm(enumerate(bags)):
        X_bag, y_bag = bag['X'], bag['y']

        pred_prevs = model.quantify(X_bag)
        
        hmae = evaluator.evaluate_from_prevalences(y_bag, pred_prevs)
        
        true_prevs_nested = evaluator.calculate_true_prevalences(y_bag)
        true_prevs_flat = {cls: prev for level_dict in true_prevs_nested.values() for cls, prev in level_dict.items()}

        resultados_batch.append({
          "bag_idx": bag_idx,
          "true_prevalences": json.dumps(true_prevs_flat),
          "pred_prevalences": json.dumps(pred_prevs),
          "hierarchical_absolute_error": hmae,
          "quantifier": quant_name,
          "classifier": clf_name,
          "strategy": config['strategy']
        })

      df_batch = pd.DataFrame(resultados_batch)
      df_batch.to_csv(results_path, mode='a', header=False, index=False)
      print(f"   [OK] {len(bags)} bags evaluated and save in results csv.")


if __name__ == '__main__':
  main()