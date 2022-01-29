from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import json
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib
import pickle


class TrainFinalModel(object):
    def __init__(self, X_data, y_data, clf):
        self.X = X_data
        self.y = y_data
        self.clf = clf
        self.scaler = StandardScaler()

    def train_model(self):
        X_down, y_down = self.downsample_training_data(self.X, self.y)
        X_down = self.scaler.fit_transform(X_down)
        self.clf.fit(X_down, y_down)
        return None

    def downsample_training_data(self, X_train, y_train):
        downsampler = RandomUnderSampler()
        X_res, y_res = downsampler.fit_resample(X_train, y_train)
        return X_res, y_res

    def get_columns(self):
        return self.X.columns.tolist()

    def save_scaler_and_model(self, path_to_save_columns, path_to_save_models, path_to_save_scaler):
        with open(path_to_save_columns, 'wb') as f:
            pickle.dump(self.get_columns(), f)

        joblib.dump(self.clf, path_to_save_models)
        joblib.dump(self.scaler, path_to_save_scaler)
        return None


start_time = time.time()


with open('hyperparameters.json') as f:
    hyperparameters = json.load(f)

# SSI
dv_name = 'SSI'

dataset = 'baseline'
X_data = pd.read_pickle('../../data/%s' % hyperparameters[dv_name][dataset]['X_data'])
y_data = pd.read_pickle('../../data/%s' % hyperparameters[dv_name][dataset]['y_data'])
ssi_clf = SVC(probability=True,
              gamma='scale',
              kernel=hyperparameters[dv_name][dataset]['kernel'],
              C=hyperparameters[dv_name][dataset]['C'])

path_to_save_columns = '../../models/%s_%s_columns.pkl' % (dv_name, dataset)
path_to_save_models = '../../models/%s_%s_model.joblib' % (dv_name, dataset)
path_to_save_scaler = '../../models/%s_%s_scaler.joblib' % (dv_name, dataset)

final_model = TrainFinalModel(X_data, y_data, ssi_clf)
final_model.train_model()
final_model.save_scaler_and_model(path_to_save_columns, path_to_save_models, path_to_save_scaler)


dataset = 'grouped'
X_data = pd.read_pickle('../../data/%s' % hyperparameters[dv_name][dataset]['X_data'])
y_data = pd.read_pickle('../../data/%s' % hyperparameters[dv_name][dataset]['y_data'])
ssi_clf = LogisticRegression(max_iter=50000,
                             solver=hyperparameters[dv_name][dataset]['solver'],
                             penalty=hyperparameters[dv_name][dataset]['penalty'],
                             C=hyperparameters[dv_name][dataset]['C'])

path_to_save_columns = '../../models/%s_%s_columns.pkl' % (dv_name, dataset)
path_to_save_models = '../../models/%s_%s_model.joblib' % (dv_name, dataset)
path_to_save_scaler = '../../models/%s_%s_scaler.joblib' % (dv_name, dataset)

final_model = TrainFinalModel(X_data, y_data, ssi_clf)
final_model.train_model()
final_model.save_scaler_and_model(path_to_save_columns, path_to_save_models, path_to_save_scaler)


end_time = time.time()
total_time = end_time - start_time
print('Total run time: %f seconds' % total_time)
