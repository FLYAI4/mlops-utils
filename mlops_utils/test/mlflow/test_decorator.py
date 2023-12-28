import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from mlops_utils.mlflow_logger.sklearn import Sklearn

sklearn = Sklearn("http://127.0.0.1", 5000)

@pytest.mark.asyncio
@sklearn.logger("test_experiment", "test_runner")
def test_runner_mlflow_logging():
    
    # prepare training data
    noise = np.random.rand(100, 1)
    X = sorted(10 * np.random.rand(100, 1)) + noise
    y = sorted(10 * np.random.rand(100))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # train a model
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    sklearn.post_board(y_test, preds)