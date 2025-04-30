# import modules
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd

def data_norm():
    # define output path
    output_folderpath = Path(__file__).resolve().parents[2] / "data" / "processed_data"

    # import raw data
    for file in output_folderpath.iterdir():
        if file.name == "X_test.csv":
            test = pd.read_csv(file)
        elif file.name == "X_train.csv":
            train = pd.read_csv(file)

    # init scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train)
    X_test_scaled = scaler.transform(test)

    # save files
    for array, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        df = pd.DataFrame(array, columns=train.columns)  # use original column names
        output_filepath = output_folderpath / f"{filename}.csv"
        df.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    data_norm()