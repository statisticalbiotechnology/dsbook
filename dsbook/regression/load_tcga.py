import pandas as pd
import tarfile
import os
import requests

def download_and_extract_tar(url, tar_path, extract_to="."):
    # Download the tar file if it does not exist
    if not os.path.exists(tar_path):
        print(f"Downloading {tar_path}...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        print("Download complete.")
    
    # Extract the tar file
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(path=extract_to)
    print(f"Files extracted to {extract_to}")

def load_table(file_path, sep="\t", drop_columns=None, index_col=None):
    # Load the file into a DataFrame
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=sep)
        if drop_columns:
            df.drop(columns=drop_columns, inplace=True, errors='ignore')
        if index_col:
            df.set_index(index_col, inplace=True)
        return df
    else:
        raise FileNotFoundError(f"The file {file_path} was not found.")

def get_expression_data(tar_path, url, file_name):
    download_and_extract_tar(url, tar_path)
    df = load_table(file_name, drop_columns=['Unnamed: 0', 'Entrez_Gene_Id'], index_col='Hugo_Symbol')
    df.dropna(axis=0, inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def get_clinical_data(tar_path, url, file_name):
    download_and_extract_tar(url, tar_path)
    df = load_table(file_name).T
    df.columns = df.iloc[2]  # Assuming that the third row contains the column names
    df.drop(columns=["A unique sample identifier.", "STRING", "1", "SAMPLE_ID"], inplace=True, errors='ignore')
    if 'TCGA-BH-A1ES-01' in df.columns:
        df.drop(columns=['TCGA-BH-A1ES-01'], inplace=True) 
    df.drop(index=["Unnamed: 0", "#Patient Identifier", "Sample Identifier", "Other Sample ID"], inplace=True, errors='ignore')
    df = df.reindex(sorted(df.columns), axis=1)
    return df

# Example usage:
# tar_path = "data.tar.gz"
# url = "https://example.com/data.tar.gz"
# expression_data = get_expression_data(tar_path, url, "data_expression.txt")
# clinical_data = get_clinical_data(tar_path, url, "data_clinical.txt")
# print(expression_data.head())
# print(clinical_data.head())

