import time
import pandas as pd
from sklearn.svm import SVC
from base_tuning import DownsampledTuning


start_time = time.time()

dv_name = 'SSI'
dataset = 'grouped'

X_data = pd.read_pickle('../../data/split_sets/X_%s_train_SSI.pkl' % dataset)
y_data = pd.read_pickle('../../data/split_sets/y_%s_train_SSI.pkl' % dataset)
X_5 = pd.read_pickle('../../data/split_sets/X_feature_selection_sets/SSI_%s_selected_5.pkl' % dataset)
X_10 = pd.read_pickle('../../data/split_sets/X_feature_selection_sets/SSI_%s_selected_10.pkl' % dataset)
X_25 = pd.read_pickle('../../data/split_sets/X_feature_selection_sets/SSI_%s_selected_25.pkl' % dataset)
X_50 = pd.read_pickle('../../data/split_sets/X_feature_selection_sets/SSI_%s_selected_50.pkl' % dataset)
X_100 = pd.read_pickle('../../data/split_sets/X_feature_selection_sets/SSI_%s_selected_100.pkl' % dataset)
X_500 = pd.read_pickle('../../data/split_sets/X_feature_selection_sets/SSI_%s_selected_500.pkl' % dataset)

base_clf = SVC(probability=True, gamma='scale')
algorithm = 'svm'
parameters = {
    'C': [0.01, 0.1, 0.25, 0.50, 1, 2],
    'kernel': ['linear', 'rbf']
}


tune = DownsampledTuning(X_data, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results(dataset, dv_name, algorithm, num_of_features=805)
print('Done with all features!')

tune = DownsampledTuning(X_5, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results(dataset, dv_name, algorithm, num_of_features=5)
print('Done with 5 features!')

tune = DownsampledTuning(X_10, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results(dataset, dv_name, algorithm, num_of_features=10)
print('Done with 10 features!')

tune = DownsampledTuning(X_25, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results(dataset, dv_name, algorithm, num_of_features=25)
print('Done with 25 features!')

tune = DownsampledTuning(X_50, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results(dataset, dv_name, algorithm, num_of_features=50)
print('Done with 50 features!')

tune = DownsampledTuning(X_100, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results(dataset, dv_name, algorithm, num_of_features=100)
print('Done with 100 features!')

tune = DownsampledTuning(X_500, y_data, base_clf, parameters, 10)
tune.tune_parameters()
tune.process_and_save_results(dataset, dv_name, algorithm, num_of_features=500)
print('Done with 500 features!')


end_time = time.time()
total_time = end_time - start_time
print('Total run time: %f seconds' % total_time)
