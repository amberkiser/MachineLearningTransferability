import pandas as pd
import numpy as np
from joblib import load
import pickle


class EvaluateModel(object):
    def __init__(self, X_data, y_data, model, scaler):
        self.X = X_data
        self.y = y_data
        self.y_pred = None
        self.y_prob = None
        self.clf = model
        self.scaler = scaler

    def run_test(self):
        X_scaled = self.scaler.transform(self.X)
        self.y_pred = self.clf.predict(X_scaled)
        y_prob = self.clf.predict_proba(X_scaled)
        self.y_prob = y_prob[:, 1]

    def save_evaluation(self, dataset):
        y_vals = pd.DataFrame({'y_true': self.y.values, 'y_pred': self.y_pred, 'y_prob': self.y_prob})
        y_vals.to_csv('../../results/validation/internal/SSI_%s_y_vals.csv' % dataset, index=False)


dataset_list = ['baseline', 'grouped']
for dataset in dataset_list:
    model = load('../../models/SSI_%s_model.joblib' % dataset)
    scaler = load('../../models/SSI_%s_scaler.joblib' % dataset)
    with open('../../models/SSI_%s_columns.pkl' % dataset, 'rb') as f:
        columns = pickle.load(f)

    X_data = pd.read_pickle('../../data/split_sets/X_%s_test_SSI.pkl' % dataset)
    X_data = X_data[columns]
    y_data = pd.read_pickle('../../data/split_sets/y_%s_test_SSI.pkl' % dataset)

    final_model = EvaluateModel(X_data, y_data, model, scaler)
    final_model.run_test()
    final_model.save_evaluation(dataset)
