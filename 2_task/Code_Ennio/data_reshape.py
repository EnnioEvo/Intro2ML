import pandas as pd
import numpy as np
import math

# Data import from folder
X_train = pd.read_csv("../data/train_features_clean_all.csv")
X_test = pd.read_csv("../data/test_features_clean_all.csv")

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

all_features = ['pid', 'Age'] + dummy_tests + \
               sum([[feature + str(i + 1) for feature in (tests + vital_signs)] for i in range(12)], [])

reordered_features = ['pid', 'Age'] + \
               sum( [[test + str(i) for i in range(1,13)] + ['dummy_'+test] for test in tests], []) +\
                sum( [[VS + str(i) for i in range(1,13)] for VS in vital_signs], [])



def column_dataset(dataset):
    # Concatenation of all the data for the 12 hours in data_set_x_train to obtain one single raw for each patient
    N = dataset.shape[0]
    X_columned = np.zeros((math.ceil(N / 12),
                           len(['pid'] + patient_characteristics + dummy_tests) + 12 * len(
                               tests + vital_signs)))

    X = np.array(dataset[['pid'] + patient_characteristics + dummy_tests + tests + vital_signs])
    for i in range(0, N, 12):
        X_temp = np.array([X[i, 0:2]])
        X_temp = np.array(
            [np.concatenate( (X_temp, np.sum([X[i:i + 12, 2:2 + len(dummy_tests)]], 1) * 1), axis=None )]
        )
        for j in range(i, i + 12):
            X_temp = np.array(
                [np.concatenate( (X_temp, np.array([ X[j, 2 + len(dummy_tests):] ])), axis=None )])
        X_columned[int(j / 12), :] = X_temp
    return X_columned

X_train_columned = column_dataset(X_train)
df_train = pd.DataFrame(X_train_columned, columns=all_features)
df_train = df_train[reordered_features]
df_train.to_csv('../data/train_features_clean_columned.csv', header=True, index=False)

X_test_columned = column_dataset(X_test)
df_test = pd.DataFrame(X_test_columned, columns=all_features)
df_test = df_test[reordered_features]
df_test.to_csv('../data/test_features_clean_columned.csv', header=True, index=False)