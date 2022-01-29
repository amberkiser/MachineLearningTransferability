import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np
from scipy import stats


def bootstrap_data(dataset):
    internal_val = pd.read_csv('../../results/validation/internal/SSI_%s_y_vals.csv' % dataset)
    external_val = pd.read_csv('../../results/validation/external/SSI_%s_y_vals.csv' % dataset)

    internal_boot = resample(internal_val, stratify=internal_val['y_true'])
    external_boot = resample(external_val, stratify=external_val['y_true'])
    return internal_boot, external_boot


def get_scores(data):
    f1 = f1_score(data['y_true'], data['y_pred'])
    auc = roc_auc_score(data['y_true'], data['y_prob'])
    acc = accuracy_score(data['y_true'], data['y_pred'])
    balanced_acc = balanced_accuracy_score(data['y_true'], data['y_pred'])
    precision = precision_score(data['y_true'], data['y_pred'])
    sensitivity = recall_score(data['y_true'], data['y_pred'])

    tn, fp, fn, tp = confusion_matrix(data['y_true'], data['y_pred']).ravel()
    specificity = tn / (tn + fp)
    return f1, auc, acc, balanced_acc, precision, sensitivity, specificity


def one_run(dataset1, dataset2):
    """
    One validation run of the models, including internal validation, external validation and
    difference in difference.

    Difference = internal validation - external validation (positive means decrease in performance,
    negative means increase in performance)

    Difference in Difference (DID) = dataset1 - dataset2 (positive means dataset2 performed better than
    dataset1)
    """

    # bootstrap data
    ds1_internal, ds1_external = bootstrap_data(dataset1)
    ds2_internal, ds2_external = bootstrap_data(dataset2)

    # get scores
    f1_ds1_internal, auc_ds1_internal, acc_ds1_internal, bacc_ds1_internal, \
    pre_ds1_internal, sen_ds1_internal, spec_ds1_internal = get_scores(ds1_internal)
    f1_ds1_external, auc_ds1_external, acc_ds1_external, bacc_ds1_external, \
    pre_ds1_external, sen_ds1_external, spec_ds1_external = get_scores(ds1_external)

    f1_ds2_internal, auc_ds2_internal, acc_ds2_internal, bacc_ds2_internal, \
    pre_ds2_internal, sen_ds2_internal, spec_ds2_internal = get_scores(ds2_internal)
    f1_ds2_external, auc_ds2_external, acc_ds2_external, bacc_ds2_external, \
    pre_ds2_external, sen_ds2_external, spec_ds2_external = get_scores(ds2_external)

    # differences
    f1_ds1_difference = f1_ds1_internal - f1_ds1_external
    f1_ds2_difference = f1_ds2_internal - f1_ds2_external
    auc_ds1_difference = auc_ds1_internal - auc_ds1_external
    auc_ds2_difference = auc_ds2_internal - auc_ds2_external
    acc_ds1_difference = acc_ds1_internal - acc_ds1_external
    acc_ds2_difference = acc_ds2_internal - acc_ds2_external
    bacc_ds1_difference = bacc_ds1_internal - bacc_ds1_external
    bacc_ds2_difference = bacc_ds2_internal - bacc_ds2_external
    pre_ds1_difference = pre_ds1_internal - pre_ds1_external
    pre_ds2_difference = pre_ds2_internal - pre_ds2_external
    sen_ds1_difference = sen_ds1_internal - sen_ds1_external
    sen_ds2_difference = sen_ds2_internal - sen_ds2_external
    spec_ds1_difference = spec_ds1_internal - spec_ds1_external
    spec_ds2_difference = spec_ds2_internal - spec_ds2_external

    # DiD
    F1_DiD = f1_ds1_difference - f1_ds2_difference
    AUC_DiD = auc_ds1_difference - auc_ds2_difference
    ACC_DiD = acc_ds1_difference - acc_ds2_difference
    BACC_DiD = bacc_ds1_difference - bacc_ds2_difference
    PRE_DiD = pre_ds1_difference - pre_ds2_difference
    SEN_DiD = sen_ds1_difference - sen_ds2_difference
    SPEC_DiD = spec_ds1_difference - spec_ds2_difference

    # make final dataframe
    run_df = pd.DataFrame({'outcome': ['SSI'], 'dataset1': [dataset1], 'dataset2': [dataset2],
                           'F1_ds1_internal': [f1_ds1_internal], 'F1_ds1_external': [f1_ds1_external],
                           'F1_ds1_difference': [f1_ds1_difference],
                           'F1_ds2_internal': [f1_ds2_internal], 'F1_ds2_external': [f1_ds2_external],
                           'F1_ds2_difference': [f1_ds2_difference],

                           'AUC_ds1_internal': [auc_ds1_internal], 'AUC_ds1_external': [auc_ds1_external],
                           'AUC_ds1_difference': [auc_ds1_difference],
                           'AUC_ds2_internal': [auc_ds2_internal], 'AUC_ds2_external': [auc_ds2_external],
                           'AUC_ds2_difference': [auc_ds2_difference],

                           'ACC_ds1_internal': [acc_ds1_internal], 'ACC_ds1_external': [acc_ds1_external],
                           'ACC_ds1_difference': [acc_ds1_difference],
                           'ACC_ds2_internal': [acc_ds2_internal], 'ACC_ds2_external': [acc_ds2_external],
                           'ACC_ds2_difference': [acc_ds2_difference],

                           'BACC_ds1_internal': [bacc_ds1_internal], 'BACC_ds1_external': [bacc_ds1_external],
                           'BACC_ds1_difference': [bacc_ds1_difference],
                           'BACC_ds2_internal': [bacc_ds2_internal], 'BACC_ds2_external': [bacc_ds2_external],
                           'BACC_ds2_difference': [bacc_ds2_difference],

                           'PRE_ds1_internal': [pre_ds1_internal], 'PRE_ds1_external': [pre_ds1_external],
                           'PRE_ds1_difference': [pre_ds1_difference],
                           'PRE_ds2_internal': [pre_ds2_internal], 'PRE_ds2_external': [pre_ds2_external],
                           'PRE_ds2_difference': [pre_ds2_difference],

                           'SEN_ds1_internal': [sen_ds1_internal], 'SEN_ds1_external': [sen_ds1_external],
                           'SEN_ds1_difference': [sen_ds1_difference],
                           'SEN_ds2_internal': [sen_ds2_internal], 'SEN_ds2_external': [sen_ds2_external],
                           'SEN_ds2_difference': [sen_ds2_difference],

                           'SPEC_ds1_internal': [spec_ds1_internal], 'SPEC_ds1_external': [spec_ds1_external],
                           'SPEC_ds1_difference': [spec_ds1_difference],
                           'SPEC_ds2_internal': [spec_ds2_internal], 'SPEC_ds2_external': [spec_ds2_external],
                           'SPEC_ds2_difference': [spec_ds2_difference],

                           'F1_DiD': [F1_DiD], 'AUC_DiD': [AUC_DiD], 'ACC_DiD': [ACC_DiD],
                           'BACC_DiD': [BACC_DiD], 'PRE_DiD': [PRE_DiD], 'SEN_DiD': [SEN_DiD],
                           'SPEC_DiD': [SPEC_DiD]})

    return run_df


def calculate_stats(results, number_of_iterations=1000):
    means = results.groupby(['outcome', 'dataset1', 'dataset2']).mean().reset_index().add_suffix('_means')
    std = results.groupby(['outcome', 'dataset1', 'dataset2']).std().reset_index().add_suffix('_std')
    std = std.replace(to_replace=0, value=0.0000000000001)

    metrics = means.merge(std, left_on=['outcome_means', 'dataset1_means', 'dataset2_means'],
                          right_on=['outcome_std', 'dataset1_std', 'dataset2_std'])
    metrics = metrics.drop(columns=['outcome_std', 'dataset1_std', 'dataset2_std'])
    metrics = metrics.rename(columns={'outcome_means': 'outcome', 'dataset1_means': 'dataset1',
                                      'dataset2_means': 'dataset2'})

    col_list = results.drop(columns=['outcome', 'dataset1', 'dataset2']).columns
    ci_dict = {}

    for column_name in col_list:
        ci_dict['%s_ci' % column_name] = []

        for i in range(len(metrics)):
            row = metrics.iloc[i]
            ci = stats.norm.interval(0.95, loc=row['%s_means' % column_name],
                                     scale=row['%s_std' % column_name] / np.sqrt(number_of_iterations))
            ci_dict['%s_ci' % column_name].append(ci)

    metrics = pd.concat([metrics, pd.DataFrame(ci_dict)], axis=1)
    return metrics
