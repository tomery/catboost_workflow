# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import shap

class Visualizer:
    def __init__(self, model, X_test, y_test, reports_path):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.reports_path = reports_path
        os.makedirs(self.reports_path, exist_ok=True)
    
    def plot_roc_curve(self):
        preds = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, preds)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.reports_path, 'roc_curve.png'))
        plt.close()
    
    def plot_confusion_matrix(self):
        preds = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.reports_path, 'confusion_matrix.png'))
        plt.close()
    
    def classification_report(self):
        preds = self.model.predict(self.X_test)
        report = classification_report(self.y_test, preds)
        with open(os.path.join(self.reports_path, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    def shap_summary_plot(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        plt.figure()
        shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False)
        plt.savefig(os.path.join(self.reports_path, 'shap_summary.png'), bbox_inches='tight')
        plt.close()
    
    def generate_all_reports(self):
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.classification_report()
        self.shap_summary_plot()
        print(f"All reports saved in {self.reports_path}")
