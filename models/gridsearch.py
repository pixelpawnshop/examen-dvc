# import modules
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
from pathlib import Path

def search():
    # path to load from
    folderpath = Path(__file__).resolve().parents[1] / "data" / "processed_data"

    # load scaled data
    X_train = pd.read_csv(folderpath / 'X_train_scaled.csv')
    y_train = pd.read_csv(folderpath / 'y_train.csv').values.ravel()

    # define model and parameter grid
    model = ElasticNet(random_state=42)
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10],         # Regularization strength
        'l1_ratio': [0.1, 0.5, 0.9, 1],      # The elasticnet mixing parameter
        'max_iter': [1000, 5000, 10000],     # Maximum number of iterations
    }

    # setup grid search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,                           # Increased cross-validation folds for better evaluation
        scoring='r2',                    # R² scoring
        n_jobs=-1,                        # Use all available cores
        verbose=1
    )

    # fit the grid search
    grid_search.fit(X_train, y_train)

    # get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # save best model parameters
    output_dir = Path('models')
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_dir / 'best_model.pkl')
    joblib.dump(best_params, output_dir / 'best_params.pkl')

if __name__ == "__main__":
    search()