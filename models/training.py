# import modules
import pandas as pd
import joblib
from sklearn.linear_model import ElasticNet
from pathlib import Path

def train():
    # path to load from
    folderpath = Path(__file__).resolve().parents[1] / "data" / "processed_data"

    # load scaled data
    X_train = pd.read_csv(f'{folderpath}/X_train_scaled.csv')
    y_train = pd.read_csv(f'{folderpath}/y_train.csv')

    # load best params
    best_params = joblib.load('models/best_params.pkl')

    # init ElasticNet model with best params
    model = ElasticNet(**best_params, random_state=42)

    # train model
    model.fit(X_train, y_train)

    # save the trained model
    output_dir = Path('models')
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / 'trained_model.pkl')

    print("model trained and saved to 'models/trained_model.pkl'")

if __name__ == "__main__":
    train()
