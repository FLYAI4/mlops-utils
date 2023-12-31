import os 
import sys

test_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
root_path = os.path.abspath(os.path.join(test_path, os.path.pardir))
mlops_utils_path = os.path.abspath(os.path.join(root_path, "mlops_utils"))
mlflow_path = os.path.abspath(os.path.join(mlops_utils_path, 'mlflow_logger'))

if mlops_utils_path not in sys.path:
    sys.path.append(mlops_utils_path)
if mlflow_path not in sys.path:
    sys.path.append(mlflow_path)

