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

# quantificationlib imports
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


def extract_hierarchy_nodes(labels):
  """Extrai todos os prefixos únicos da hierarquia."""
  nodes = set()
  for label in labels:
    parts = label.split('.')
    for i in range(1, len(parts) + 1):
        nodes.add('.'.join(parts[:i]))
  return sorted(list(nodes))


def extract_embeddings(df):
  X = [eval(e) if isinstance(e, str) else e for e in df['embeddings'].values]
  return np.stack(X) if len(np.array(X).shape) == 1 else np.array(X)


def train_lqn(X, labels, quant_name, base_clf):
  """Treina um quantificador binário independente para CADA NÓ da hierarquia."""
  all_nodes = extract_hierarchy_nodes(labels)
  models = {}

  for node in all_nodes:
    # Gera labels binários: 1 se pertence ao nó ou é descendente, 0 caso contrário
    # O startswith(node + '.') garante que 'A.B' não capture 'A.BC' por engano
    y_bin = np.array([
      1 if (lbl == node or lbl.startswith(node + '.')) else 0 
      for lbl in labels
    ])

    # Se o nó abrange todo o dataset (ex: raiz global) ou não tem instâncias, ignoramos
    if len(np.unique(y_bin)) < 2:
      print(f"Treino LQN - Nó '{node}' ignorado: não há variância binária (só possui 1 classe).")
      continue

    print(f"Treinando LQN - Nó: {node} (Positivos: {y_bin.sum()} / Total: {len(y_bin)})")
    if quant_name in ['em', 'ac', 'dfy', 'hdy', 'edy', 'cvmy']:
      clf = clone(base_clf)
      quantifier = QUANTIFIERS[quant_name](estimator_train=clf, estimator_test=clf)
    else:
      quantifier = QUANTIFIERS[quant_name](clone(base_clf))

    quantifier.fit(X, y_bin)
    models[node] = quantifier

  return models


def generate_hierarchical_bags(df, n_bags, bag_size, alpha):
  """Gera bags utilizando o protocolo Dirichlet APP."""
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
            
    # O verdadeiro diferencial: calculamos a prevalência exata de cada nó 
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
  parser = argparse.ArgumentParser(description="Pipeline LQN (Local Quantifier per Node)")
  parser.add_argument("--config", type=str, default="config_lqn.yaml", help="Caminho para o arquivo config.yaml")
  args = parser.parse_args()
  
  print(f"Carregando configurações LQN de: {args.config}")
  cfg = load_config(args.config)
  
  data_cfg = cfg['data']
  model_cfg = cfg['model']
  app_cfg = cfg['app_protocol']

  set_seed(app_cfg['seed'])

  # --- ETAPA 1: TREINO (LQN) ---
  print(f"\n--- Iniciando Treino LQN ({model_cfg['quantifier'].upper()} + {model_cfg['classifier'].upper()}) ---")
  df_train = pd.read_parquet(data_cfg['train_data'], engine='pyarrow')
  X_train = extract_embeddings(df_train)
  labels_train = df_train['label'].values
  base_clf = CLASSIFIERS[model_cfg['classifier']]
  
  models = train_lqn(X_train, labels_train, model_cfg['quantifier'], base_clf)

  # --- ETAPA 2: GERAÇÃO DE BAGS (TESTE) ---
  print(f"\n--- Iniciando Geração de Teste ---")
  df_test = pd.read_parquet(data_cfg['test_data'], engine='pyarrow')
  print(f"Gerando {app_cfg['n_bags']} bags (Tamanho: {app_cfg['bag_size']}, Alpha: {app_cfg['alpha']})...")
  bags = generate_hierarchical_bags(df_test, n_bags=app_cfg['n_bags'], bag_size=app_cfg['bag_size'], alpha=app_cfg['alpha'])

  # --- ETAPA 3: INFERÊNCIA E CÁLCULO DE ERRO ---
  print(f"\n--- Iniciando Inferência ---")
  results = []
  
  for idx, bag in enumerate(bags):
    bag_df = bag['bag_df']
    # true_prevs já contém a prevalência real de todos os nós gerados pela bag
    true_prevs_hierarchical = bag['true_prevalences'] 
    
    X_bag = extract_embeddings(bag_df)
    pred_prevs = {}
    
    # Iterando por cada nó treinado
    for node, model in models.items():
      estimations = model.predict(X_bag)
      
      # Resgatando as classes mapeadas internamente (tipicamente [0, 1])
      classes = list(getattr(model, 'classes_', getattr(getattr(model, 'estimator', None), 'classes_', [0, 1])))
      
      # Queremos a prevalência prevista da classe positiva (1)
      if 1 in classes:
        idx_1 = classes.index(1)
        pred_prevs[node] = float(estimations[idx_1])
      else:
        pred_prevs[node] = 0.0

    # Cálculo do Mean Absolute Error (MAE) agregado para a bag usando todos os nós
    # Interseção das chaves para garantir que não vamos perder nenhum nó real ou predito
    all_keys = set(pred_prevs.keys()).union(set(true_prevs_hierarchical.keys()))
    
    abs_errors = []
    for node in all_keys:
      p_pred = pred_prevs.get(node, 0.0)
      p_true = true_prevs_hierarchical.get(node, 0.0)
      abs_errors.append(abs(p_pred - p_true))
        
    bag_ae = np.mean(abs_errors) if abs_errors else 0.0

    # Salvar resultado da bag
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
  df_results.to_csv(data_cfg['output'], index=False)
  
  mean_ae_all_bags = df_results['absolute_error'].mean()
  print(f"\n✅ Sucesso! Resultados salvos em: {data_cfg['output']}")