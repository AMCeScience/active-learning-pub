import numpy as np
from sklearn.metrics import confusion_matrix

class PerformanceObj(object):
  def __init__(self):
    self.new_iteration(0, 0)


  def new_iteration(self, total_documents, total_includes):
    # Metrics to be recorded for each iteration in active learning
    self.yield_list = []
    self.burden_list = []

    self.total_documents = total_documents
    self.total_includes = total_includes


  def unpack_confusion_matrix(self, confusion_matrix):
    # If only one element left in the confusion matrix, then ravel() cannot be used, so here divide it into two paths
    if len(confusion_matrix) > 1:
      _, FP, FN, TP = confusion_matrix.ravel()
    else:
      FP = 0
      FN = 0
      TP = 0

    return FP, FN, TP


  def calculate_performance(self, clf, X_test, y_test, tp_labelled, reviewer_list):
    num_read = len(reviewer_list)

    predictions = clf.predict(X_test)

    # Get confusion matrix for the test data
    cm = confusion_matrix(y_test, predictions)

    # Calculate Yield and Burden
    cur_yield = self.get_yield(cm, tp_labelled)
    cur_burden = self.get_burden(cm, self.total_documents, num_read)

    self.yield_list.append(cur_yield)
    self.burden_list.append(cur_burden)


  def get_performance_object(self):
    # Create a dictionary to store the results
    result_obj = {
      'yield': self.yield_list,
      'burden': self.burden_list
    }

    return result_obj


  def get_yield(self, confusion_matrix, tp_labelled):
    FP, FN, TP = self.unpack_confusion_matrix(confusion_matrix)

    return (tp_labelled + TP) / (tp_labelled + TP + FN)


  def get_burden(self, confusion_matrix, N, num_read):
    FP, FN, TP = self.unpack_confusion_matrix(confusion_matrix)

    return ((num_read + TP + FP) / N)