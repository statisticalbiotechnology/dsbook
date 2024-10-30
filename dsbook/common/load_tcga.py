import pandas as pd
import tarfile
import shutil
import os
import requests


def download_and_extract_gz(url, gz_path, extract_to="../data/"):
    # Download the gz file if it does not exist
    if not os.path.exists(gz_path):
        print(f"Downloading {gz_path}...")
        response = requests.get(url, stream=True)
        with open(gz_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        print("Download complete.")
    
    # Ensure the extract_to directory exists
    os.makedirs(extract_to, exist_ok=True)

    with tarfile.open(gz_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)

    # Determine the extracted file name and path
    extracted_file_name = os.path.splitext(os.path.splitext(os.path.basename(gz_path))[0])[0]
    extracted_file_path = os.path.join(extract_to, extracted_file_name)
        
    print(f"File extracted to {extracted_file_path}")
    return extracted_file_path

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


def load_table(file_path, sep="\t", drop_columns=None, index_col=None, skiprows=None):
    # Load the file into a DataFrame
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=sep, skiprows=skiprows)
        if drop_columns:
            df.drop(columns=drop_columns, inplace=True, errors='ignore')
        if index_col:
            df.set_index(index_col, inplace=True)
        return df
    else:
        raise FileNotFoundError(f"The file {file_path} was not found.")

def get_expression_data(gz_path, url, file_name):
    extracted_file = download_and_extract_gz(url, gz_path)
    df = load_table(''.join([extracted_file,'/', file_name]), drop_columns=['Entrez_Gene_Id'], index_col='Hugo_Symbol')
    df.dropna(axis=0, inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def get_clinical_data(gz_path, url, file_name):
    extracted_file = download_and_extract_gz(url, gz_path)
    df = load_table(''.join([extracted_file,'/', file_name]), index_col="SAMPLE_ID", skiprows=4).T
    df.drop(columns=["A unique sample identifier.", "STRING", "1"], inplace=True, errors='ignore')
    if 'TCGA-BH-A1ES-01' in df.columns:
        df.drop(columns=['TCGA-BH-A1ES-01'], inplace=True) 
    df.drop(index=["Unnamed: 0", "#Patient Identifier", "Sample Identifier", "Other Sample ID"], inplace=True, errors='ignore')
    df = df.reindex(sorted(df.columns), axis=1)
    return df

# Example usage:
# gz_path = "data_expression.txt.gz"
# url = "https://example.com/data_expression.txt.gz"
# expression_data = get_expression_data(gz_path, url, "data_expression.txt")
# clinical_data = get_clinical_data(gz_path, url, "data_clinical.txt")
# print(expression_data.head())
# print(clinical_data.head())

