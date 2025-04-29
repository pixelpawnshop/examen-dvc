# import modules
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd

# import raw data
for file in folder_path.iterdir():
    if file.name == "X_test.csv":
        test = pd.read_csv(file)  # file already includes the full path
    elif file.name == "X_train.csv":
        train = pd.read_csv(file)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)