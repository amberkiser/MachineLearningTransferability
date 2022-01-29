import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


class ANOVAFeatureSelection(object):
    def __init__(self, X_data, y_data, dv_name, num_of_features):
        self.X = X_data
        self.X_norm = None
        self.y = y_data
        self.num_of_features = num_of_features
        self.feature_selector = SelectKBest(f_classif, k=num_of_features)
        self.dv_name = dv_name
        self.results = None

    def scale_data(self):
        scaler = StandardScaler()
        self.X_norm = scaler.fit_transform(self.X)
        return None

    def find_top_features(self):
        self.feature_selector.fit(self.X_norm, self.y)
        return None

    def save_features(self, results_path):
        self.results = pd.DataFrame({'feature': self.X.columns, 'scores': self.feature_selector.scores_,
                                     'pvalues': self.feature_selector.pvalues_,
                                     'support': self.feature_selector.get_support()})
        self.results = self.results.loc[self.results['support']]
        self.results.to_csv('%s/%s_feature_selection_%d.csv' % (results_path, self.dv_name, self.num_of_features))
        return None

    def save_dataset(self, dataset_path):
        self.X = self.X[self.results['feature'].values]
        self.X.to_pickle('%s/%s_selected_%d.pkl' % (dataset_path, self.dv_name, self.num_of_features))
        return None


# Load data
X_baseline = pd.read_pickle('../data/split_sets/X_baseline_train_SSI.pkl')
y_baseline = pd.read_pickle('../data/split_sets/y_baseline_train_SSI.pkl')

X_grouped = pd.read_pickle('../data/split_sets/X_grouped_train_SSI.pkl')
y_grouped = pd.read_pickle('../data/split_sets/y_grouped_train_SSI.pkl')

# Paths for storing feature selection results
results_path = '../results/feature_selection'
dataset_path = '../data/split_sets/X_feature_selection_sets'

feature_number_list = [5, 10, 25, 50, 100, 500, 1000, 5000]
for i in feature_number_list:
    feature_selection = ANOVAFeatureSelection(X_baseline, y_baseline, 'SSI_baseline', num_of_features=i)
    feature_selection.scale_data()
    feature_selection.find_top_features()
    feature_selection.save_features(results_path)
    feature_selection.save_dataset(dataset_path)

feature_number_list = [5, 10, 25, 50, 100, 500]
for i in feature_number_list:
    feature_selection = ANOVAFeatureSelection(X_grouped, y_grouped, 'SSI_grouped', num_of_features=i)
    feature_selection.scale_data()
    feature_selection.find_top_features()
    feature_selection.save_features(results_path)
    feature_selection.save_dataset(dataset_path)
