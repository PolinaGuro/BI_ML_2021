import numpy as np

def binary_classification_metrics(Mod, Exp):

  tp = 0; fp = 0; fn = 0; tn = 0
  for i in range(Mod.shape[0]):
    if Mod[i] == 0 and Exp[i] == 0:
      tn = tn + 1
    elif Mod[i] == 0 and Exp[i] == 1:
      fn = fn + 1
    elif Mod[i] == 1 and Exp[i] == 0:
      fp = fp + 1
    else:
      tp = tp + 1
  if not (tp + fp == 0):
    precision = tp / (tp + fp)
  else:
    precision = 0
  if not (tp + fn == 0):  
    recall = tp / (tp + fn)
  else:
    recall = 0
  if not (precision + recall == 0):  
    F1 = 2 * (precision * recall) / (precision + recall)
  else:
    F1 = 0
  if not (tp + fn + fp + tn == 0):  
    accuracy = (tp + tn) / (tp + fn + fp + tn)
  else:
    accuracy = 0

  return precision, recall, F1, accuracy
  pass


def multiclass_accuracy(Mod, Exp):
  num_Classes = np.size(np.unique(Exp))
  arr = np.zeros((num_Classes, num_Classes))
  for i in range(np.size(Mod, 0)):
    arr[np.int16(Mod[i]), np.int16(Exp[i])] = arr[np.int16(Mod[i]), np.int16(Exp[i])] + 1
  precision = np.zeros(num_Classes)
  for i in range(np.size(precision, 0)):
    if not (np.sum(arr[i, :]) == 0):
      precision[i] = arr[i,i] / np.sum(arr[i, :])
    else:
      precision[i] = 0
      
  recall = np.zeros(num_Classes)
  for i in range(np.size(recall, 0)):
    if not (np.sum(arr[:, i]) == 0):
      recall[i] = arr[i,i] / np.sum(arr[:, i])
    else:
      recall[i] = 0
  return precision
  pass


def r_squared(y_pred, y_true):
  mean_y_true = np.mean(y_true)
  r_sq = 1 - np.sum((y_pred - y_true)**2)/np.sum((y_true - mean_y_true)**2)
  pass


def mse(y_pred, y_true):
  mse = np.sum((y_pred - y_true)**2)/y_true.shape[0]
  pass


def mae(y_pred, y_true):
  mae = np.sum(np.abs(y_pred - y_true))/y_true.shape[0]
  pass
