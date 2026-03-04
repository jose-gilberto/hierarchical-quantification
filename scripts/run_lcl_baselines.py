import argparse
import pandas as pd
import numpy as np
import json
import random
import yaml

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# quantificationlib imports (Certifique-se de que a biblioteca está instalada/acessível)
from quantificationlib.baselines.cc import CC
from quantificationlib.baselines.ac import AC
from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.df import DFy, HDy
from quantificationlib.multiclass.energy import EDy, CvMy


CLASSIFIERS = {
  'rf': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
  'lr': LogisticRegression(max_iter=1000, random_state=42),
  'svm': SVC(probability=True, random_state=42)
}

QUANTIFIERS = {
  'cc': CC,
  'ac': AC,
  'em': EM,
  'dfy': DFy,
  'hdy': HDy,
  'edy': EDy,
  'cvmy': CvMy,
}


def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)


def inject_root_node(labels):
  """Garante que todas as labels comecem com 'Root.'"""
  return [f"Root.{lbl}" if not lbl.startswith("Root") else lbl for lbl in labels]


def parse_hierarchical_labels(labels):
  return [label.split('.') for label in labels]


def get_level_labels(parsed_labels, level):
  return [path[level] if level < len(path) else path[-1] for path in parsed_labels]


def extract_hierarchy_nodes(labels):
  nodes = set()
  for label in labels:
    parts = label.split('.')
    for i in range(1, len(parts) + 1):
      nodes.add('.'.join(parts[:i]))
  return sorted(list(nodes))


def extract_embeddings(df):
  X = [eval(e) if isinstance(e, str) else e for e in df['embeddings'].values]
  return np.stack(X) if len(np.array(X).shape) == 1 else np.array(X)


def train_lcl(X, parsed_labels, quant_name, base_clf):
  max_depth = max(len(path) for path in parsed_labels)
  models = {}

  for lvl in range(max_depth): 
    y_lvl = get_level_labels(parsed_labels, lvl)
    classes = np.unique(y_lvl)

    if len(classes) < 2:
      print(f"Treino LCL - Nível {lvl} ignorado: apenas uma classe presente.")
      continue

    print(f"Treinando LCL - Nível {lvl} ({len(classes)} classes)...")
    if quant_name in ['em', 'ac', 'dfy', 'hdy', 'edy', 'cvmy']:
      clf = clone(base_clf)
      quantifier = QUANTIFIERS[quant_name](estimator_train=clf, estimator_test=clf)
    else:
      quantifier = QUANTIFIERS[quant_name](clone(base_clf))
    quantifier.fit(X, y_lvl)
    models[lvl] = quantifier

  return models


def generate_hierarchical_bags(df, n_bags, bag_size, alpha):
  df_internal = df.copy().reset_index(drop=True)
  base_classes = df_internal['label'].unique()
  all_nodes = extract_hierarchy_nodes(base_classes)
  
  base_prevalences = np.random.dirichlet(np.ones(len(base_classes)) * alpha, n_bags)
  bags_data = []
  
  for prev_vector in base_prevalences:
    bag_indices = []
    counts = np.round(prev_vector * bag_size).astype(int)
    
    diff = bag_size - counts.sum()
    if diff != 0:
      counts[np.argmax(counts)] += diff
        
    for cls, count in zip(base_classes, counts):
      if count > 0:
        cls_indices = df_internal[df_internal['label'] == cls].index
        sampled = np.random.choice(cls_indices, size=count, replace=True)
        bag_indices.extend(sampled)
          
    hierarchical_prevs = {node: 0.0 for node in all_nodes}
    for cls, count in zip(base_classes, counts):
      parts = cls.split('.')
      for i in range(1, len(parts) + 1):
        node = '.'.join(parts[:i])
        hierarchical_prevs[node] += count / bag_size
            
    bag_df = df_internal.loc[bag_indices].copy().sample(frac=1).reset_index(drop=True)
    
    bags_data.append({
      'bag_df': bag_df,
      'true_prevalences': hierarchical_prevs
    })
      
  return bags_data


def load_config(config_path):
  with open(config_path, 'r') as file:
    return yaml.safe_load(file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Pipeline Unificado via YAML com AE")
  parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para o arquivo config.yaml")
  args = parser.parse_args()
  
  print(f"Carregando configurações de: {args.config}")
  cfg = load_config(args.config)
  
  data_cfg = cfg['data']
  model_cfg = cfg['model']
  app_cfg = cfg['app_protocol']

  set_seed(app_cfg['seed'])

  # --- ETAPA 1: TREINO ---
  print(f"--- Iniciando Treino ({model_cfg['quantifier'].upper()} + {model_cfg['classifier'].upper()}) ---")
  df_train = pd.read_parquet(data_cfg['train_data'], engine='pyarrow')
  df_train['label'] = inject_root_node(df_train['label'].values)

  X_train = extract_embeddings(df_train)
  parsed_labels_train = parse_hierarchical_labels(df_train['label'].values)
  base_clf = CLASSIFIERS[model_cfg['classifier']]
  
  models = train_lcl(X_train, parsed_labels_train, model_cfg['quantifier'], base_clf)

  # --- ETAPA 2: GERAÇÃO DE BAGS (TESTE) ---
  print(f"\n--- Iniciando Geração de Teste ---")
  df_test = pd.read_parquet(data_cfg['test_data'], engine='pyarrow')
  df_test['label'] = inject_root_node(df_test['label'].values)

  print(f"Gerando {app_cfg['n_bags']} bags (Tamanho: {app_cfg['bag_size']}, Alpha: {app_cfg['alpha']})...")
  bags = generate_hierarchical_bags(df_test, n_bags=app_cfg['n_bags'], bag_size=app_cfg['bag_size'], alpha=app_cfg['alpha'])

  # --- ETAPA 3: INFERÊNCIA E CÁLCULO DE ERRO ---
  print(f"\n--- Iniciando Inferência ---")
  results = []
  
  for idx, bag in enumerate(bags):
    bag_df = bag['bag_df']
    true_prevs_hierarchical = bag['true_prevalences']
    X_bag = extract_embeddings(bag_df)
    
    pred_prevs = {}
    true_prevs_mapped = {}
    
    parsed_bag_labels = parse_hierarchical_labels(bag_df['label'].values)
    bag_size_actual = len(bag_df)
    max_depth = max(len(path) for path in parsed_bag_labels)
    
    # 1. Mapear a prevalência REAL para todos os níveis (incluindo o Nível 0 / Root)
    for lvl in range(max_depth):
      y_true_lvl = get_level_labels(parsed_bag_labels, lvl)
      unique_classes, counts = np.unique(y_true_lvl, return_counts=True)
      for cls, count in zip(unique_classes, counts):
        true_prevs_mapped[f"L{lvl}_{cls}"] = count / bag_size_actual

    # 2. Injetar a predição manual do Root (Nível 0)
    pred_prevs["L0_Root"] = 1.0
    
    # 3. Pegar predições reais dos modelos treinados (Nível 1 em diante)
    for lvl, model in models.items():
      estimations = model.predict(X_bag)
      classes = getattr(model, 'classes_', getattr(getattr(model, 'estimator', None), 'classes_', None))
      
      if classes is not None:
        for cls, prev in zip(classes, estimations):
          pred_prevs[f"L{lvl}_{cls}"] = float(prev)

    # 4. Cálculo do Mean Absolute Error (MAE) alinhando as chaves
    all_keys = set(pred_prevs.keys()).union(set(true_prevs_mapped.keys()))
    abs_errors = [abs(pred_prevs.get(k, 0.0) - true_prevs_mapped.get(k, 0.0)) for k in all_keys]
    bag_ae = np.sum(abs_errors) if abs_errors else 0.0

    results.append({
      'bag_idx': idx,
      'true_prevalence': json.dumps(true_prevs_hierarchical),
      'pred_prevalence': json.dumps(pred_prevs),
      'absolute_error': bag_ae,
      'quantifier': model_cfg['quantifier'],
      'classifier': model_cfg['classifier'],
      'strategy': model_cfg['strategy']
    })

  # --- ETAPA 4: SALVAR RESULTADOS ---
  df_results = pd.DataFrame(results)
  df_results.to_csv(f'{data_cfg['output']}/{model_cfg['strategy']}_{model_cfg['quantifier']}_{model_cfg['classifier']}.csv', index=False)
  
  # Exibe um resumo rápido do erro médio no terminal
  mean_ae_all_bags = df_results['absolute_error'].mean()
  print(f"\n✅ Sucesso! Resultados salvos em: {data_cfg['output']}")
  print(f"📊 Erro Absoluto Médio (MAE) de todas as bags: {mean_ae_all_bags:.4f}")