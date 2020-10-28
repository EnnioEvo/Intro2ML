import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
import copy

np.random.seed(seed=277)
# from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------
# ------------ DATA IMPORT AND DEFINITIONS ----------------
# ---------------------------------------------------------

# cleaned data import:
train_features = pd.read_csv("../data/train_features_clean_mean_diff.csv")
test_features = pd.read_csv("../data/test_features_clean_mean_diff.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
sample = pd.read_csv("../sample.csv")
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
# ----------------- SET PARAMETERS ------------------------
# ---------------------------------------------------------
epochs = 3
margin = 1e-4
examinated_features = standard_features + diff_features

# ---------------------------------------------------------
# ----------------- DATA SELECTION ------------------------
# ---------------------------------------------------------

def build_set(selected_features, train_size):

    # Definition of test and val data size:
    # task 1
    X = train_features.loc[0:train_size - 1, selected_features]
    X_val = train_features.loc[train_size:, selected_features]
    X_test = test_features[selected_features]

    # # add dummy features
    # X = np.column_stack([X, np.array(train_features.loc[0:train_size - 1, dummy_tests])])
    # X_val = np.column_stack([X_val, np.array(train_features.loc[train_size:, dummy_tests])])
    # X_test = np.column_stack([X_test, np.array(test_features[dummy_tests])])

    # Standardize the data
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    X_val = (X_val - np.mean(X_val, 0)) / np.std(X_val, 0)
    X_test = (X_test - np.mean(X_test, 0)) / np.std(X_test, 0)

    X[np.isnan(X)] = 0
    X_val[np.isnan((X_val))] = 0
    X_test[np.isnan(X_test)] = 0

    return X, X_val, X_test


# --------------------------------------------------------
# ------------------- TRAINING TASK 1 --------------------
# ---------------------------------------------------------
usefulness_matrixes_t1 = []
usefulness_matrix_t1_sum = 0
labels_target = labels_tests + ['LABEL_Sepsis']
train_size = int(train_features.shape[0] * 0.8)
selected_features_t1 = standard_features + dummy_tests + diff_features
for k in range(epochs):
    # shuffle
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))

    X_t1, X_val_t1, X_test_t1 = build_set(selected_features_t1, train_size)
    usefulness_matrix_t1 = pd.DataFrame(index=examinated_features, columns=labels_target)
    scores_t1 = []
    for i in range(0, len(labels_target)):
        label_target = labels_target[i]
        Y_t1 = train_labels[label_target].iloc[0:train_size]
        Y_val_t1 = train_labels[label_target].iloc[train_size:]

        # fit
        clf = svm.LinearSVC(C=1e-4, class_weight='balanced', tol=10e-2, verbose=0)
        clf.fit(X_t1, Y_t1)
        # predict
        Y_temp = np.array([clf.decision_function(X_val_t1)])
        Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
        # score
        score = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred)])
        print("Removed: Nothing", "\nscore ", label_target, " :", score)
        score = 0

        useful_features_mask = np.repeat([False], len(examinated_features))
        useful_features_mask[-1] = True
        for _ in range(10):
            useful_features_mask_temp = useful_features_mask
            for j in range(0, len(examinated_features)):
                # build a new mask
                new_useful_features_mask = copy.deepcopy(useful_features_mask)
                new_useful_features_mask[j] = ~new_useful_features_mask[j]
                useful_features = [feature for feature, mask in zip(examinated_features, new_useful_features_mask) if
                                   mask]
                useful_features_augmented = sum(
                    [[feature, 'dummy_' + feature] for feature in useful_features if feature in tests], []) + \
                                            [feature for feature in useful_features if feature in vital_signs + diff_features] \
                                            # + sum([sum(
                                            #     [[feature + suffix] for feature in useful_features if
                                            #      feature in vital_signs],
                                            #     []) for suffix in diff_features_suffixes], [])
                X_t1_useful = X_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
                X_val_t1_useful = X_val_t1[list(set(useful_features_augmented) & set(X_t1.columns))]

                # fit
                clf = svm.LinearSVC(C=1e-4, class_weight='balanced', tol=10e-2, verbose=0)
                clf.fit(X_t1_useful, Y_t1)

                # predict
                Y_temp = np.array([clf.decision_function(X_val_t1_useful)])
                Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()

                # score
                new_score = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred)])
                #print("Removed: ", examinated_features[j], "\nscore ", label_target, " :", new_score)
                #print()
                if new_score > score + margin:
                    score = new_score
                    useful_features_mask = copy.deepcopy(new_useful_features_mask)
            if np.all(
                    useful_features_mask_temp == useful_features_mask):  # if the list of useful features has not changed exit
                break


        usefulness_matrix_t1[label_target] = useful_features_mask
        scores_t1 = scores_t1 + [score]
        print("ROC AUC final score ", i, " ", label_target, " :", score, '\n')

        # mask with found useful features
        useful_features = [feature for feature, mask in zip(examinated_features, new_useful_features_mask) if
                           mask]
        useful_features_augmented = sum(
            [[feature, 'dummy_' + feature] for feature in useful_features if feature in tests], []) \
                                    + [feature for feature in useful_features if feature in vital_signs + diff_features] \
            # + sum([sum(
        #     [[feature + suffix] for feature in useful_features if
        #      feature in vital_signs],
        #     []) for suffix in diff_features_suffixes], [])
        X_t1_useful = X_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
        X_val_t1_useful = X_val_t1[list(set(useful_features_augmented) & set(X_t1.columns))]

        # last fit for current label
        clf = svm.LinearSVC(C=1e-4, class_weight='balanced', tol=10e-2, verbose=0)
        clf.fit(X_t1_useful, Y_t1)


    task1 = sum(scores_t1[:-1]) / len(scores_t1[:-1])
    print("ROC AUC task1 score  ", task1)
    task2 = scores_t1[-1]
    print("ROC AUC task2 score ", task2)
    usefulness_matrix_t1_sum = usefulness_matrix_t1_sum + (usefulness_matrix_t1 == 0) * -1 + (
            usefulness_matrix_t1 == 1) * 1
usefulness_matrix_t1_sum.to_csv('../data/usefulness_matrix_t1_sum.csv', header=True, index=True, float_format='%.1f')

# ---------------------------------------------------------
# ------------------- TRAINING TASK 3 --------------------
# ---------------------------------------------------------

usefulness_matrixes_t3 = []
usefulness_matrix_t3_sum = 0

labels_target = labels_VS_mean
for k in range(epochs):
    # shuffle
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))

    train_size = int(train_features.shape[0] * 0.8)
    selected_features_t3 = standard_features + dummy_tests + diff_features
    X_t3, X_val_t3, X_test_t3 = build_set(selected_features_t3, train_size)

    usefulness_matrix_t3 = pd.DataFrame(index=examinated_features, columns=labels_target)
    scores_t3 = []
    for i in range(0, len(labels_target)):
        # get the set corresponding tu the feature
        label_target = labels_target[i]
        Y_t3 = train_labels[label_target].iloc[0:train_size]
        Y_val_t3 = train_labels[label_target].iloc[train_size:]

        # fit
        reg = LinearRegression()
        reg.fit(X_t3, Y_t3)
        # reg = Lasso(alpha=0.01)
        # reg.fit(X_train_subtask3, np.ravel(Y_train))

        # score
        Y_val_pred = reg.predict(X_val_t3).flatten()
        score = 0.5 + 0.5 * skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')
        print("R2 score initial features ", i, " ", label_target, " :", score)
        score = 0
        useful_features_mask = np.repeat([False], len(examinated_features))
        useful_features_mask[-1] = True
        for _ in range(10):
            useful_features_mask_temp = useful_features_mask
            for j in range(len(examinated_features)):
                # build a new mask
                new_useful_features_mask = copy.deepcopy(useful_features_mask)
                new_useful_features_mask[j] = ~new_useful_features_mask[j]
                useful_features = [feature for feature, mask in zip(examinated_features, new_useful_features_mask) if
                                   mask]
                useful_features_augmented = sum(
                    [[feature, 'dummy_' + feature] for feature in useful_features if feature in tests], []) \
                                            + [feature for feature in useful_features if feature in vital_signs+ diff_features] \
                                            # + sum([sum(
                                            #     [[feature + suffix] for feature in useful_features if
                                            #      feature in vital_signs],
                                            #     []) for suffix in diff_features_suffixes], [])
                X_t3_useful = X_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
                X_val_t3_useful = X_val_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
                X_test_t3_useful = X_test_t3[list(set(useful_features_augmented) & set(X_t3.columns))]


                # fit
                reg = LinearRegression()
                reg.fit(X_t3_useful, Y_t3)

                # predict
                Y_val_pred = reg.predict(X_val_t3_useful).flatten()

                # score
                new_score = 0.5 + 0.5 * skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')
                #print("Removed: ", standard_features[j], "\nscore ", label_target, " :", new_score)
                if new_score > score + margin:
                    score = new_score
                    useful_features_mask = copy.deepcopy(new_useful_features_mask)
            if np.all(
                    useful_features_mask_temp == useful_features_mask):  # if the list of useful features has not changed exit
                break
        # upload usefulness matrix
        usefulness_matrix_t3[label_target] = useful_features_mask

        # print score
        scores_t3 = scores_t3 + [score]
        print("ROC AUC final score ", i, " ", label_target, " :", score, '\n')

        # mask with found useful features
        useful_features = [feature for feature, mask in zip(standard_features, new_useful_features_mask) if
                           mask]
        useful_features_augmented = sum(
            [[feature, 'dummy_' + feature] for feature in useful_features if feature in tests], []) + \
                                    [feature for feature in useful_features if feature in vital_signs + diff_features] \
                                    # + sum([sum(
                                    #     [[feature + suffix] for feature in useful_features if
                                    #      feature in vital_signs],
                                    #     []) for suffix in diff_features_suffixes], [])
        X_t3_useful = X_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
        X_val_t3_useful = X_val_t3[list(set(useful_features_augmented) & set(X_t3.columns))]

        # last fit for current label
        reg = LinearRegression()
        reg.fit(X_t3_useful, Y_t3)

    scores_t3 = [score_t3 for score_t3 in scores_t3]

    print("Task 3 single scores: ")
    for score_t3 in scores_t3:
        print(score_t3, "\n")

    task3 = np.mean(scores_t3)
    print("Task3 total score: ", task3)
    # print("Total score = ", np.mean([task1, task2, task3]))
    usefulness_matrix_t3_sum = usefulness_matrix_t3_sum + (usefulness_matrix_t3 == 0) * -1 + (
            usefulness_matrix_t3 == 1) * 1


usefulness_matrix_t3_sum.to_csv('../data/usefulness_matrix_t3_sum.csv', header=True, index=True, float_format='%.1f')


