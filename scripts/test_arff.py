import pandas as pd
import arff
import numpy as np
import re

# Substitua pelo caminho do seu arquivo local
FILE_PATH = './data/datasets_FUN/church_FUN/church_FUN.train.arff'

def load_clus_arff(filepath):
    print(f"Lendo arquivo bruto: {filepath}...")
    
    # 1. Ler o conteúdo como texto puro
    # Usamos iso-8859-1 para garantir que não quebre em acentos
    with open(filepath, 'r', encoding='iso-8859-1') as f:
        content = f.read()
    
    # 2. Corrigir o cabeçalho "hierarchical"
    # O Regex abaixo procura a linha que começa com "@ATTRIBUTE class hierarchical" 
    # e a substitui por "@ATTRIBUTE class string"
    # Isso faz o parser tratar a coluna de classes como um texto comum.
    content_fixed = re.sub(
        r'@ATTRIBUTE\s+class\s+hierarchical.*', 
        '@ATTRIBUTE class string', 
        content, 
        flags=re.IGNORECASE
    )
    
    print("Cabeçalho corrigido na memória. Convertendo para DataFrame...")
    
    # 3. Carregar o ARFF a partir da string corrigida
    dataset = arff.load(content_fixed)
    
    attributes = [attr[0] for attr in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=attributes)
    
    return df

# Executar
df = load_clus_arff(FILE_PATH)

print("--- Sucesso! ---")
print(df.head())
print("\nVerificando a coluna de classe (agora é string):")
# Geralmente é a última coluna
print(df.iloc[:5, -1])

def process_hierarchical_labels(df):
    # Identificar a coluna de classe (geralmente chamada 'class')
    class_col = 'class' 
    if class_col not in df.columns:
        # Pega a última se não achar pelo nome
        class_col = df.columns[-1]
    
    print(f"Processando coluna de rótulos: '{class_col}'")
    
    # 1. Criar um conjunto de todas as classes possíveis
    # As classes no Clus costumam ser separadas por '@' ou ',' dentro da string
    all_labels = set()
    
    # Função para limpar e separar os labels de uma linha
    def parse_labels(value):
        if not value or pd.isna(value): 
            return []
        # Remove espaços e quebra por separadores comuns (@ ou ,)
        # No formato Clus, labels múltiplos são frequentemente unidos por @
        parts = re.split(r'[@,]', str(value))
        return [p.strip() for p in parts if p.strip()]

    # Aplicar a função para extrair listas de labels
    df['labels_list'] = df[class_col].apply(parse_labels)
    
    # Coletar universo de labels únicos
    for labels in df['labels_list']:
        all_labels.update(labels)
        
    print(f"Total de classes únicas encontradas: {len(all_labels)}")
    
    # 2. (Opcional) Criar colunas binárias para cada classe (One-Hot Encoding Manual)
    # Isso cria o formato X (features) e Y (matriz de classes)
    
    # Cuidado: Se houver 4000 classes, isso criará 4000 colunas!
    # Vamos criar apenas para as Top 10 para visualização rápida, 
    # ou use MultiLabelBinarizer do sklearn para tudo.
    
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Y_matrix = mlb.fit_transform(df['labels_list'])
    Y_df = pd.DataFrame(Y_matrix, columns=mlb.classes_)
    
    # Retorna as features originais (sem a coluna 'class') e a matriz Y
    X = df.drop(columns=[class_col, 'labels_list'])
    
    return X, Y_df

# Uso
X, Y = process_hierarchical_labels(df)

import matplotlib.pyplot as plt
import pandas as pd

def analyze_levels(Y):
    print("--- Análise de Níveis Hierárquicos ---\n")
    
    # Dicionário para armazenar classes por nível
    # Chave = Nível (int), Valor = Lista de tuplas (nome_da_classe, frequencia)
    levels = {}
    
    # Calcular a frequência de cada classe para exibir junto
    class_counts = Y.sum().to_dict()
    
    for label in Y.columns:
        # A profundidade é o número de barras '/' + 1
        # Ex: "01" (0 barras) -> Nível 1
        # Ex: "01/02" (1 barra) -> Nível 2
        depth = label.count('/') + 1
        
        if depth not in levels:
            levels[depth] = []
            
        levels[depth].append((label, class_counts[label]))
    
    # Ordenar e exibir
    total_classes = 0
    level_stats = [] # Para o gráfico
    
    sorted_depths = sorted(levels.keys())
    
    for depth in sorted_depths:
        classes_in_level = levels[depth]
        count = len(classes_in_level)
        total_classes += count
        
        # Ordenar as classes deste nível pelas mais frequentes
        classes_in_level.sort(key=lambda x: x[1], reverse=True)
        
        print(f"=== NÍVEL {depth} ===")
        print(f"Total de classes: {count}")
        
        # Mostrar exemplos (Top 5 mais frequentes deste nível)
        top_examples = classes_in_level[:5]
        example_str = ", ".join([f"{name} ({freq})" for name, freq in top_examples])
        print(f"Exemplos mais comuns: {example_str} ...")
        print("-" * 40)
        
        level_stats.append(count)

    print(f"\nTotal Geral de Classes: {total_classes}")
    print(f"Profundidade Máxima: {max(sorted_depths)}")

    # --- Visualização: Quantidade de Classes por Nível ---
    plt.figure(figsize=(10, 5))
    bars = plt.bar(sorted_depths, level_stats, color='#4c72b0', edgecolor='black')
    
    plt.title('Quantidade de Classes por Nível Hierárquico')
    plt.xlabel('Nível (Profundidade)')
    plt.ylabel('Número de Classes Distintas')
    plt.xticks(sorted_depths)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar rótulos nas barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')
        
    plt.show()

# Executar a função (assumindo que Y é seu DataFrame binário)
analyze_levels(Y)

# Agora você pode usar as funções de visualização anteriores passando Y
# visualize_data(Y) (função definida na resposta anterior)