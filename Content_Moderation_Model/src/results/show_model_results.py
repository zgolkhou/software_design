import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


def print_confusion_matrix(path):
  """Prints confusion matrix of model test results
  Parameters:
  -----------
  path: path to predictions csv file
  multiclass: whether the confusion matrix is for a binary or multiclass model

  Returns
  Prints confusion matrix plot"""

  # Import predictions data
  test_df_preds = pd.read_csv(path)

  # Confusion matrix
  LABELS = [ "Non-Viol","Viol" ]
  cm = tf.math.confusion_matrix(test_df_preds["ViolationBool"], 
                              test_df_preds['Prediction'])

  # Normalize the confusion matrix so that each row sums to 1.
  cm = tf.cast(cm, dtype=tf.float32)
  cm = cm / tf.math.reduce_sum(cm, axis=1)[:, np.newaxis]

  # Print confusion matrix
  sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS);
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.show()
  plt.savefig('../reports/figures/binary_confusion_matrix.png')


def show_classification_report(path):
  """Prints classification report of model test results
    Input: Path to predictions data
    Output: Prints classification report"""

  # Import predictions data
  test_df_preds = pd.read_csv(path)

  print('\nClassification Report\n')
  # The numerical lebels are correct, but the English labels are mostly incorrect, I'm waiting to receive a corrected key.
  print(classification_report(test_df_preds['ViolationBool'], test_df_preds['Prediction']))

def show_multiclassification_report(path):
  """Prints classification report of model test results
    Input: Path to predictions data
    Output: Prints classification report"""

  # Import predictions data
  performance_df = pd.read_csv(path)

  print('\nMulti Classification Report\n')
  print(classification_report(performance_df['Violation_Category'],performance_df['Prediction_Category']))


def plot_roc_curve(path):
  """Plots roc curve of model test results. For use only with binary classification.
    Parameters:
    -----------
    path: path to predictions csv file
    multiclass: whether the confusion matrix is for a binary or multiclass model

    Returns:
    --------
    Prints confusion matrix plot"""

  # Import predictions data
  test_df_preds = pd.read_csv(path)
  fpr, tpr, thresholds = roc_curve(test_df_preds["ViolationBool"], test_df_preds['Probability'])
  

  plt.plot(fpr, tpr, color='orange', label='ROC')
  plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend()
  plt.savefig('../reports/figures/roc_curve.png')


def print_auc_score(path, multiclass=False):
  """Prints AUC score of model test results. For use only with binary classification.
    Input: Path to predictions data
    Output: Prints classification report"""
  
  # Import predictions data
  test_df_preds = pd.read_csv(path)

  auc = roc_auc_score(test_df_preds["ViolationBool"], test_df_preds['Probability'])
  print('AUC: %.2f' % auc)

