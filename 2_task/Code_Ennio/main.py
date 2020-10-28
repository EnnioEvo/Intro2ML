import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from threading import Lock, Thread

np.random.seed(seed=377)
# from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# ------------ DATA IMPORT AND DEFINITIONS ----------------
# ---------------------------------------------------------

# cleaned data import:
train_features = pd.read_csv("../data/train_features_clean_mean_diff.csv")
test_features = pd.read_csv("../data/test_features_clean_mean_diff.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
sample = pd.read_csv("../sample.csv")
stored_usefulness_matrix_t1 = pd.read_csv("../data/usefulness_matrix_t1_sum.csv", index_col=0)
best_kernels = pd.read_csv("../data/best_kernels.csv", index_col=0)

# features
patient_characteristics = ["Age"]  # TIME VARIABLE IS EXCLUDED
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate", 'Temp']
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess',
         'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
         'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
         'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
         'Bilirubin_total', 'TroponinI', 'pH']
dummy_tests = ['dummy_' + test for test in tests]

standard_features = patient_characteristics + vital_signs + tests
diff_features_suffixes = ['_n_extrema', '_diff_mean', '_diff_median', '_diff_max', '_diff_min']
diff_features = sum(
    [[VS + diff_features_suffix for VS in vital_signs] for diff_features_suffix in diff_features_suffixes], [])
all_features = standard_features + dummy_tests + diff_features

# labels
labels_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                'LABEL_EtCO2']
labels_sepsis = ['LABEL_Sepsis']
labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
all_labels = labels_tests + labels_sepsis + labels_VS_mean

# Drop pid feature:
train_features = train_features.drop(labels="pid", axis=1)
test_features = test_features.drop(labels="pid", axis=1)

# ---------------------------------------------------------
# ----------------- SET GENERAL PARAMETERS ----------------
# ---------------------------------------------------------
submit = False
shuffle = True
remove_outliers = True
parallel = True

# ---------------------------------------------------------
# ----------------- SET PARAMETERS TASK 1------------------
# ---------------------------------------------------------
classifier = 'linear'  # choose between 'linear', 'kernel' and 'RF'
use_diff = True
features_selection = False
threshold = -2


# ---------------------------------------------------------
# ----------------- DATA SELECTION ------------------------
# ---------------------------------------------------------
def _get_percentiles(data_set, min, max):
    perc = pd.DataFrame(columns=data_set.columns)
    perc.loc[0, :] = np.nanpercentile(np.array(data_set), min,
                                      axis=0, interpolation='lower')
    perc.loc[1, :] = np.nanpercentile(np.array(data_set), max,
                                      axis=0, interpolation='higher')
    return perc


if remove_outliers:
    percentiles = _get_percentiles(train_features, 0.01, 99.99)
    percentiles = percentiles[tests]
    for feature in percentiles.columns:
        mask = np.multiply(
            train_features[feature] > percentiles[feature][0],
            train_features[feature] < percentiles[feature][1])
        train_features = train_features[mask]
        train_labels = train_labels[mask]

if shuffle:
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))


def build_set(selected_features, train_size, submit):
    # Definition of test and val data size:
    # task 1
    if submit:
        X = train_features[selected_features].iloc[:]
        X_val = train_features[selected_features].iloc[train_size:]
        X_test = test_features[selected_features]
    else:
        X = train_features[selected_features].iloc[0:train_size - 1]
        X_val = train_features[selected_features].iloc[train_size:]
        X_test = test_features[selected_features]

    # Standardize the data
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    X_val = (X_val - np.mean(X_val, 0)) / np.std(X_val, 0)
    X_test = (X_test - np.mean(X_test, 0)) / np.std(X_test, 0)

    # Set NaN to 0
    X[np.isnan(X)] = 0
    X_val[np.isnan((X_val))] = 0
    X_test[np.isnan(X_test)] = 0

    return X, X_val, X_test


# Build sets
# task 1
train_size = int(train_features.shape[0] * 0.8)
selected_features_t1 = standard_features + dummy_tests
if use_diff:
    selected_features_t1 = selected_features_t1 + diff_features
# selected_features_t2 = vital_signs + diff_features
X_t1, X_val_t1, X_test_t1 = build_set(selected_features_t1, train_size, submit)

# Variable for storing prediction
Y_test_tot = pd.DataFrame(np.zeros([X_test_t1.shape[0], len(all_labels)]),
                          columns=all_labels)  # predictions for test set
Y_val_tot = pd.DataFrame(np.zeros([X_val_t1.shape[0], len(all_labels)]), columns=all_labels)  # predictions for val set
Y_test_tot.insert(0, 'pid', sample['pid'])
# --------------------------------------------------------
# ------------------- TRAINING TASK 1 --------------------
# ---------------------------------------------------------

labels_target = labels_tests + ['LABEL_Sepsis']
scores_t1 = pd.DataFrame(columns=labels_target, index=[0])
lock = Lock()


def process_target_t1(label_target):
    if submit:
        Y_t1 = train_labels[label_target].iloc[:]
        Y_val_t1 = train_labels[label_target].iloc[train_size:]
    else:
        Y_t1 = train_labels[label_target].iloc[0:train_size - 1]
        Y_val_t1 = train_labels[label_target].iloc[train_size:]

    if features_selection:
        usefulness_column = stored_usefulness_matrix_t1[label_target].sort_values(ascending=False)
        useful_features_mask = np.array(usefulness_column) >= threshold
        useful_features = [feature for feature, mask in zip(usefulness_column.index, useful_features_mask) if mask]
        useful_features_augmented = sum(
            [[test, 'dummy_' + test] for test in useful_features if test in tests], []) \
                                    + [feature for feature in useful_features if feature in vital_signs + diff_features] \
            # + sum([sum(
        #     [[feature + suffix] for feature in useful_features if feature in vital_signs],
        #     []) for suffix in diff_features_suffixes], [])
        X_t1_useful = X_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
        X_val_t1_useful = X_val_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
        X_test_t1_useful = X_test_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
    else:
        X_t1_useful = X_t1
        X_val_t1_useful = X_val_t1
        X_test_t1_useful = X_test_t1

    # fit

    if classifier == 'linear' or (classifier == 'kernel' and best_kernels.at[label_target, 'kernel'] == 'poly1'):
        clf = svm.LinearSVC(C=1e-3, tol=1e-2, class_weight='balanced', verbose=0)
    elif classifier == 'kernel':
        kernel_dict = {'poly2': ('poly', 2), 'poly3': ('poly', 3), 'rbf': ('rbf', 0)}
        kernel, degree = kernel_dict[best_kernels.at[label_target, 'kernel']]
        C = best_kernels.at[label_target, 'C']
        clf = svm.SVC(C=C, kernel=kernel, degree=degree, tol=1e-4, class_weight='balanced', verbose=0)
    elif classifier == 'RF':
        clf = RandomForestClassifier(n_estimators=2500, class_weight="balanced_subsample")
    else:
        raise ValueError("choose between 'linear', 'classifier' and 'RF' ")

    # fit
    clf.fit(X_t1_useful, Y_t1)

    # predict and save into dataframe
    if classifier == 'linear' or classifier == 'kernel':
        Y_temp = np.array([clf.decision_function(X_val_t1_useful)])
        Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
        Y_temp = np.array([clf.decision_function(X_test_t1_useful)])
        Y_test_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
    elif classifier == 'RF':
        Y_val_pred = (1 - clf.predict_proba(X_val_t1_useful))[:, 0]
        Y_test_pred = (1 - clf.predict_proba(X_test_t1_useful))[:, 0]

    lock.acquire()
    Y_test_tot.loc[:, label_target] = Y_test_pred
    score = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred)])
    scores_t1[label_target] = score
    lock.release()
    return score


if parallel:
    threads = []
    for label_target in labels_target:
        threads = threads + [Thread(target=process_target_t1, args=(label_target,), name='Thread_' + label_target)]
        threads[-1].start()

    for thread in threads:
        thread.join()
else:
    for label_target in labels_target:
        process_target_t1(label_target)

for label_target in labels_target:
    print("ROC AUC -- score ", label_target, " :", scores_t1[label_target][0])

task1 = sum(scores_t1.iloc[0, :-1]) / len(scores_t1.iloc[0, :-1])
print("ROC AUC task1 score  ", task1)
task2 = scores_t1.iloc[0, -1]
print("ROC AUC task2 score ", task2)

# -------------------------------------
# ---------- BEGIN MAIN_ALL -----------
# -------------------------------------


# ---------------------------------------------------------
# ------------ DATA IMPORT AND DEFINITIONS ----------------
# ---------------------------------------------------------

# cleaned data import:
train_features = pd.read_csv("../data/train_features_clean_columned_diff.csv")
test_features = pd.read_csv("../data/test_features_clean_columned_diff.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
stored_usefulness_matrix_t3 = pd.read_csv("../data/usefulness_matrix_t3_sum.csv", index_col=0)

N_hours_test = 1
N_hours_VS = 9
houred_features = ['Age'] + \
                  sum([[test + str(i) for i in range(13 - N_hours_test, 13)] + ['dummy_' + test] for test in tests],
                      []) + \
                  sum([[VS + str(i) for i in range(13 - N_hours_VS, 13)] for VS in vital_signs], [])

all_features = patient_characteristics + vital_signs + tests + diff_features

# Drop pid feature:
train_features = train_features.drop(labels="pid", axis=1)
test_features = test_features.drop(labels="pid", axis=1)
# ---------------------------------------------------------
# ----------------- SET PARAMETERS T3 ------------------------
# ---------------------------------------------------------
regressor = 'linear'  # choose between 'linear', and 'RF'
use_diff = True
features_selection = True
threshold = -2

# ---------------------------------------------------------
# ----------------- DATA SELECTION T3------------------------
# ---------------------------------------------------------
if remove_outliers:
    percentiles = _get_percentiles(train_features, 1e-3, 100 - 1e-3)
    percentiles = percentiles[
        sum([[houred_test for houred_test in houred_features if (test in houred_test and 'dummy' not in houred_test)]
             for test in tests], [])]
    for feature in percentiles.columns:
        mask = np.multiply(
            train_features[feature] > percentiles[feature][0],
            train_features[feature] < percentiles[feature][1])
        train_features = train_features[mask]
        train_labels = train_labels[mask]

if shuffle:
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))

# task3
selected_houred_features_t3 = houred_features
if use_diff:
    selected_houred_features_t3 = selected_houred_features_t3 + diff_features
X_t3, X_val_t3, X_test_t3 = build_set(selected_houred_features_t3, train_size, submit)

Y_test_tot.to_csv('../data/submission.csv', header=True, index=False, float_format='%.7f')
Y_test_tot.to_csv('../data/submission.zip', header=True, index=False, float_format='%.7f', compression='zip')

# ---------------------------------------------------------
# ------------------- TRAINING TASK 3 --------------------
# ---------------------------------------------------------

labels_target = labels_VS_mean
# labels_target = ['LABEL_' + select_feature for select_feature in select_features]
scores_t3 = pd.DataFrame(columns=labels_target, index=[0])
lock = Lock()


def process_target_t3(label_target):
    # get the set corresponding tu the feature
    if submit:
        Y_t3 = train_labels[label_target].iloc[:]
        Y_val_t3 = train_labels[label_target].iloc[train_size:]
    else:
        Y_t3 = train_labels[label_target].iloc[0:train_size - 1]
        Y_val_t3 = train_labels[label_target].iloc[train_size:]

    if features_selection:
        usefulness_column = stored_usefulness_matrix_t3[label_target].sort_values(ascending=False)
        useful_features_mask = np.array(usefulness_column) >= threshold
        useful_features = [feature for feature, mask in zip(usefulness_column.index, useful_features_mask) if mask]
        useful_features_augmented = \
            sum([[feature for useful_feature in useful_features if useful_feature in feature] for feature in houred_features], []) \
            + [feature for feature in useful_features if feature in diff_features] \

            # + sum([sum(
        #     [[feature + suffix] for feature in useful_features if feature in vital_signs],
        #     []) for suffix in diff_features_suffixes], [])
        X_t3_useful = X_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
        X_val_t3_useful = X_val_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
        X_test_t3_useful = X_test_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
    else:
        X_t3_useful = X_t3
        X_val_t3_useful = X_val_t3
        X_test_t3_useful = X_test_t3

    # fit
    if regressor == 'linear':
        #reg = LinearRegression()
        reg = Ridge(alpha=100)
    elif regressor == 'RF':
        reg = RandomForestRegressor(n_estimators=330)
    else:
        raise ValueError("choose between 'linear' and 'RF' ")

    # fit
    reg.fit(X_t3_useful, np.ravel(Y_t3))

    # predict
    if regressor == 'linear':
        Y_test_pred = reg.predict(X_test_t3_useful).flatten()
        Y_val_pred = reg.predict(X_val_t3_useful).flatten()
    elif regressor == 'RF':
        Y_val_pred = reg.predict(X_val_t3_useful)
        Y_test_pred = reg.predict(X_test_t3_useful)

    score = 0.5 + 0.5 * skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')

    # save into dataframe
    lock.acquire()
    Y_test_tot.loc[:, label_target] = Y_test_pred
    scores_t3[label_target] = score
    lock.release()


if parallel:
    threads = []
    for label_target in labels_target:
        threads = threads + [Thread(target=process_target_t3, args=(label_target,), name='Thread_' + label_target)]
        threads[-1].start()

    for thread in threads:
        thread.join()
else:
    for label_target in labels_target:
        process_target_t3(label_target)

for label_target in labels_target:
    print("R2 score ", label_target, " :", scores_t3[label_target][0])

task3 = np.mean(scores_t3.iloc[0, :])
print("Task3 score = ", task3)

print("Total score = ", np.mean([task1, task2, task3]))

# -------------------------------------
# -----------SAVE----------------------
# -------------------------------------

# save into file
Y_test_tot.to_csv('../data/submission.csv', header=True, index=False, float_format='%.7f')
Y_test_tot.to_csv('../data/submission.zip', header=True, index=False, float_format='%.7f', compression='zip')
