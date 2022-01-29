from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler


class DownsampledTuning(object):
    def __init__(self, X_data, y_data, base_clf, grid_param, cv_num):
        self.X = X_data
        self.y = y_data
        self.base_clf = base_clf
        self.parameter_combos = ParameterGrid(grid_param)
        self.k_folds = StratifiedKFold(n_splits=cv_num)
        self.cv_scores = pd.DataFrame()

    def tune_parameters(self):
        for i in range(len(self.parameter_combos)):
            self.run_cv(i)

    def run_cv(self, iteration):
        fold = 0
        for train_index, val_index in self.k_folds.split(self.X, self.y):
            fold += 1
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            X_train, y_train = self.downsample_training_data(X_train, y_train)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            clf = clone(self.base_clf)
            clf.set_params(**self.parameter_combos[iteration])
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            y_train_prob = clf.predict_proba(X_train)
            y_train_prob = y_train_prob[:, 1]
            train_scores = self.evaluate_cv_results(y_train, y_train_pred, y_train_prob, iteration)

            y_val_pred = clf.predict(X_val)
            y_test_prob = clf.predict_proba(X_val)
            y_test_prob = y_test_prob[:, 1]
            val_scores = self.evaluate_cv_results(y_val, y_val_pred, y_test_prob, iteration)

            eval_df = self.create_scores_dataframe(train_scores, val_scores, fold)
            self.cv_scores = pd.concat([self.cv_scores, eval_df])

        return None

    def downsample_training_data(self, X_train, y_train):
        downsampler = RandomUnderSampler()
        X_res, y_res = downsampler.fit_resample(X_train, y_train)
        return X_res, y_res

    def evaluate_cv_results(self, y_true, y_pred, y_prob, iteration):
        scores = {'Parameter_combo': [], 'Acc': [], 'F1': [], 'AUC': []}

        scores['Parameter_combo'].append(self.parameter_combos[iteration])
        scores['Acc'].append(accuracy_score(y_true, y_pred))
        scores['F1'].append(f1_score(y_true, y_pred))
        scores['AUC'].append(roc_auc_score(y_true, y_prob))

        return scores

    def create_scores_dataframe(self, train_dict, val_dict, fold):
        train_df = pd.DataFrame(train_dict)
        train_df['dataset'] = 'train'
        train_df['fold'] = fold

        val_df = pd.DataFrame(val_dict)
        val_df['dataset'] = 'val'
        val_df['fold'] = fold
        eval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        return eval_df

    def process_and_save_results(self, dataset, dv_name, algorithm, num_of_features):
        self.cv_scores['Parameter_combo'] = self.cv_scores['Parameter_combo'].apply(lambda x: str(x))
        results_agg = self.cv_scores.drop(columns='fold').groupby(['dataset', 'Parameter_combo']).agg('mean').reset_index()

        results_train = results_agg.loc[results_agg.dataset == 'train'].drop(columns='dataset')
        new_names = {}
        for col in results_train.columns:
            if col != 'Parameter_combo':
                new_names[col] = col + '_train'
        results_train = results_train.rename(columns=new_names)

        results_val = results_agg.loc[results_agg.dataset == 'val'].drop(columns='dataset').reset_index(drop=True)
        new_names = {}
        for col in results_val.columns:
            if col != 'Parameter_combo':
                new_names[col] = col + '_val'
        results_val = results_val.rename(columns=new_names)
        results_val['Acc_rank'] = results_val['Acc_val'].rank(axis=0, method='dense', ascending=False)
        results_val['F1_rank'] = results_val['F1_val'].rank(axis=0, method='dense', ascending=False)
        results_val['AUC_rank'] = results_val['AUC_val'].rank(axis=0, method='dense', ascending=False)

        results_combined = results_train.merge(results_val, on='Parameter_combo')
        results_combined = results_combined[['Parameter_combo', 'Acc_val', 'Acc_rank', 'F1_val', 'F1_rank', 'AUC_val',
                                             'AUC_rank', 'Acc_train', 'F1_train', 'AUC_train']]
        results_combined.to_csv('../../results/tuning/individual_results/%s/%s_%s_%d.csv' % (dataset, dv_name, algorithm,
                                                                                             num_of_features))
        return None
