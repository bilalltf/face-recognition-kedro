from sklearn import metrics

class ModelMetrics:
    def __init__(self, model):
        self.model = model
    
    def calculate_metrics(self, x, y, target_names):
        classifier = self.model.classifier
        y_pred = classifier.predict(x)
        report = metrics.classification_report(y, y_pred, target_names=target_names, output_dict=True)

        return report