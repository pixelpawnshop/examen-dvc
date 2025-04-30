# import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def data_splitting():
    # load dataset
    raw = pd.read_csv("https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv")

    # feature / target 
    X = raw.drop(["silica_concentrate", "date"], axis=1)
    y = raw["silica_concentrate"]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define output path
    output_folderpath = Path(__file__).resolve().parents[2] / "data" / "processed_data"

    # save files
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = output_folderpath / f"{filename}.csv"
        file.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    data_splitting()