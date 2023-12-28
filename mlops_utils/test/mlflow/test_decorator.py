import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlflow_logger.decorator import MlflowLogger

mlflow_logger = MlflowLogger("http://127.0.0.1", 5000)

@pytest.mark.asyncio
@mlflow_logger.sklearn("test_experiment", "test_runner")
def test_runner_mlflow_logging():
    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # train a model
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X, y)
    