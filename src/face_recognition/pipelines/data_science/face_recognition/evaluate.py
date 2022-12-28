import numpy as np
from sklearn import metrics

class ModelMetrics:
    def __init__(self, model):
        self.model = model
    
    def calculate_metrics(self, x, y, target_names):
        classifier = self.model.classifier
        probas = classifier.predict_proba(x)
        y_pred = classifier.predict(x)
        print(y_pred)
        print(y)
        report = metrics.classification_report(y_true = y, y_pred = y_pred, target_names=target_names, output_dict=True)

        return report