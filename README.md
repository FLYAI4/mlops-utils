# MLOps Utils
mlops-utils is a collection of tools for managing MLOps infrastructure.

<br>

## Installation
Install using via PyPi


```bash
pip install mlops-utils
```

<br>

## Requirements
Python 3.10 or higher is required.
For automatic tracking in PyTorch logging, versions between 1.9.0 and 2.1.1 (inclusive) are supported

<br>

## Documentation
Documentation will be added soon. Stay tuned for updates!


### Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from mlops_utils.mlflow_logger.sklearn import Sklearn

# URL 주소 변경
sklearn = Sklearn("{url}", 5000)
experiment_name = "minjun_researcher"
run_name = "20240116-sklearn"


@sklearn.logger(experiment_name, run_name)
def ml_run():
    noise = np.random.rand(100, 1)
    X = sorted(10 * np.random.rand(100, 1)) + noise
    y = sorted(10 * np.random.rand(100))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(preds)

    sklearn.post_board(y_test, preds)

ml_run()

```


