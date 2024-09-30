import os
import pandas as pd

def clean_data(filepath: str):
    print(f"Carregando dados de: {filepath}")
    # Carregar dados
    try:
        df = pd.read_csv(filepath)
        print("Dados carregados com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo {filepath} não foi encontrado.")
        return
    except pd.errors.EmptyDataError:
        print("Erro: O arquivo CSV está vazio.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao ler o CSV: {e}")
        return

    # Verificar se as colunas existem
    required_columns = ["price_brl", "volume"]
    print(f"Verificando colunas necessárias: {required_columns}")
    for col in required_columns:
        if col not in df.columns:
            print(f"Erro: A coluna '{col}' está ausente no CSV.")
            return
    print("Todas as colunas necessárias estão presentes.")

    # Remover valores nulos e normalizar dados
    print("Removendo valores nulos...")
    df = df.dropna()
    print(f"Número de linhas após dropna: {len(df)}")
    try:
        df["price_brl"] = df["price_brl"].astype(float)
        df["volume"] = df["volume"].astype(float)
        print("Conversão de tipos de dados bem-sucedida.")
    except ValueError as e:
        print(f"Erro ao converter tipos de dados: {e}")
        return

    # Salvar dados limpos
    clean_filepath = filepath.replace("raw", "processed")
    print(f"Caminho do arquivo limpo: {clean_filepath}")
    clean_dir = os.path.dirname(clean_filepath)
    print(f"Criando diretório: {clean_dir}")
    try:
        os.makedirs(clean_dir, exist_ok=True)
        print("Diretório criado ou já existente.")
    except Exception as e:
        print(f"Erro ao criar diretório {clean_dir}: {e}")
        return

    try:
        print("Salvando arquivo CSV limpo...")
        df.to_csv(clean_filepath, index=False)
        print("Arquivo CSV limpo salvo com sucesso.")
    except Exception as e:
        print(f"Ocorreu um erro ao salvar o CSV: {e}")
        return

    return clean_filepath

if __name__ == "__main__":
    input_filepath = "../../data/raw/crypto_prices/render_token_weekly_brl.csv"
    print(f"Diretório de trabalho atual: {os.getcwd()}")
    clean_filepath = clean_data(input_filepath)
    if clean_filepath:
        print(f"Dados limpos salvos em: {clean_filepath}")
