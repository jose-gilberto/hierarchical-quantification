import argparse
import arff
import pandas as pd
import re
import sys
from sklearn.preprocessing import MultiLabelBinarizer

def load_clus_arff(filepath):
    print(f"Lendo arquivo bruto: {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Corrige o cabeçalho "hierarchical" na memória
    content_fixed = re.sub(
        r'@ATTRIBUTE\s+class\s+hierarchical.*', 
        '@ATTRIBUTE class string', 
        content, 
        flags=re.IGNORECASE
    )
    
    print("Cabeçalho corrigido. Convertendo para DataFrame...")
    dataset = arff.load(content_fixed)
    attributes = [attr[0] for attr in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=attributes)
    
    return df


def parse_labels_with_ancestors(value):
    """
    Expande os rótulos para incluir todos os nós ancestrais (True Path Rule).
    """
    if not value or pd.isna(value): 
        return []
    
    parts = re.split(r'[@,]', str(value))
    raw_labels = [p.strip() for p in parts if p.strip()]
    
    full_labels = set()
    for label in raw_labels:
        path_nodes = label.split('/')
        for i in range(1, len(path_nodes) + 1):
            ancestor = '/'.join(path_nodes[:i])
            full_labels.add(ancestor)
            
    return list(full_labels)


def process_hierarchical_labels(df):
    class_col = 'class'
    if class_col not in df.columns:
        class_col = df.columns[-1]
    
    print(f"Processando e expandindo coluna de rótulos: '{class_col}'")
    
    # 1. Cria a lista real de labels (Excelente para PyTorch/Keras)
    df['labels_list'] = df[class_col].apply(parse_labels_with_ancestors)
    
    print("Aplicando MultiLabelBinarizer e otimizando memória (uint8)...")
    mlb = MultiLabelBinarizer()
    Y_matrix = mlb.fit_transform(df['labels_list'])
    
    # 2. Transforma em DataFrame usando uint8 para máxima economia de RAM e Disco
    Y_df = pd.DataFrame(Y_matrix, columns=mlb.classes_, dtype='uint8')
    
    print(f"Total de classes únicas após expansão: {len(mlb.classes_)}")
    
    # 3. Separa features (X) e mantém a labels_list para flexibilidade
    X = df.drop(columns=[class_col])
    
    return X, Y_df

def main():
    parser = argparse.ArgumentParser(
        description="Converte dataset ARFF hierárquico em um arquivo Parquet otimizado para Machine Learning."
    )
    parser.add_argument(
        '-i', '--dataset', 
        required=True, 
        help="Nome do dataset."
    )
    
    parser.add_argument(
        '-s', '--split',
        required=True,
        help='Split do dataset para ser utilizado'
    )

    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help="Caminho para o arquivo .parquet de saída."
    )
    
    args = parser.parse_args()
    
    # Substitua pelo caminho do seu arquivo local
    FILE_PATH = f'./data/multi-label/{args.dataset}/{args.dataset}_FUN.{args.split}.arff'
    
    # Garantir que a saída tenha a extensão correta
    if not args.output.endswith('.parquet'):
        print("⚠️ Aviso: O arquivo de saída não termina em .parquet. Adicionando a extensão automaticamente.")
        args.output += '.parquet'
    
    try:
        df = load_clus_arff(FILE_PATH)
        
        class_col = 'class' if 'class' in df.columns else df.columns[-1]
        
        print("Aplicando True Path Rule e convertendo para listas...")
        df['target_labels'] = df[class_col].apply(parse_labels_with_ancestors)
        
        X, Y = process_hierarchical_labels(df)
        
        print(f"Concatenando features e matriz de classes...")
        df = df.drop(columns=[class_col])
        
        print(f"Salvando resultado otimizado em: {args.output}...")
        # engine='pyarrow' lida perfeitamente com listas e tipos uint8
        df.to_parquet(args.output, engine='pyarrow', index=False)
        
        print("\n✅ Conversão concluída com sucesso!")
        print(f"Shape final do dataset: {df.shape[0]} linhas e {df.shape[1]} colunas.")
        
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo de entrada '{args.input}' não foi encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()