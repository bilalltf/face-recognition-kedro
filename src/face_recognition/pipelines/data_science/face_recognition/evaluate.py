from typing import Dict, List
from sklearn import metrics
import numpy as np

class ModelMetrics:
    def __init__(self, model):
        self.model = model
    
def calculate_metrics(self, x, y, metric_names: List[str]) -> Dict[str, float]:
    y_pred = self.model.predict(x)

    # Initialize the idx_to_class dictionary and target_names list using the test data
    idx_to_class = {i: class_name for i, class_name in enumerate(np.unique(y))}
    target_names = list(map(lambda i: idx_to_class[i], np.unique(y)))

    # Generate the classification report
    report = metrics.classification_report(y, y_pred, target_names=target_names, output_dict=True)

    # Return the metrics as a dictionary
    return {metric: report[metric] for metric in metric_names}