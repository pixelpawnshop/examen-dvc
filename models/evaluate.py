import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

def eval():
    # path to load from
    folderpath = Path(__file__).resolve().parents[1] / "data" / "processed_data"

    # load scaled data
    X_test = pd.read_csv(f'{folderpath}/X_test_scaled.csv')
    y_test = pd.read_csv(f'{folderpath}/y_test.csv').values.ravel()

    # load ElasticNet model
    model = joblib.load('models/trained_model.pkl')

    # make predictions
    y_pred = model.predict(X_test)

    # compute metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    predictions_df.to_csv('data/predictions.csv', index=False)

    # save metrics
    metrics = {
        'mean_squared_error': mse,
        'r2_score': r2
    }

    metrics_dir = Path('metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / 'scores.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # print results
    print("✅ Model evaluated.")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    eval()
