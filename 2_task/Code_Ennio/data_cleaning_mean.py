import numpy as np
import pandas as pd

# Data import
train_features = pd.read_csv("../data/train_features.csv")
test_features = pd.read_csv("../data/test_features.csv")

# Informatons on the headers
patient_characteristics = ["pid", "Age"]  # TIME VARIABLE IS EXCLUDED
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate", 'Temp']
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess',
         'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
         'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
         'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
         'Bilirubin_total', 'TroponinI', 'pH']


def nanweightedmean(array, weights):
    if len(array) != len(weights):
        raise Exception("Lenght of array and weight must match: " + str(len(array)) + ' != ' + str(len(weights)))

    array = np.array(array)
    weights = np.array(weights)

    consistent_array = array[~np.isnan(array)]
    consistent_weights = weights[~np.isnan(array)]

    return np.dot(consistent_array, consistent_weights) / sum(consistent_weights)


def cleaning(data_set):
    # initialize new dataset
    new_columns = patient_characteristics + sum([[test, 'dummy_' + test] for test in tests], []) + vital_signs
    data_set_new = pd.DataFrame(index=list(set(data_set['pid'])), columns=new_columns)

    # pid and Age
    N_patients = np.array(data_set.shape[0] / 12).astype(int)
    data_set_new[patient_characteristics] = np.array(data_set[patient_characteristics])[::12]

    # vital signs
    for VS in vital_signs:
        VS_column = np.array(data_set[VS])
        VS_mean = np.nanmean(VS_column)
        new_VS_column = np.zeros([N_patients, 1])
        for i in range(N_patients):
            if np.all(np.isnan(VS_column[i * 12:(i + 1) * 12])):
                new_VS_column[i] = VS_mean
            else:
                new_VS_column[i] = nanweightedmean(VS_column[i * 12:(i + 1) * 12], np.arange(1, 13))
        data_set_new[VS] = new_VS_column

    # tests
    for test in tests:
        test_column = np.array(data_set[test])
        test_mean = np.nanmean(test_column)
        new_test_column = np.zeros([N_patients, 1])
        dummy_test_column = np.zeros([N_patients, 1])
        for i in range(N_patients):
            if np.all(np.isnan(test_column[i * 12:(i + 1) * 12])):
                new_test_column[i] = test_mean
                dummy_test_column[i] = 0
            else:
                array_i = test_column[i * 12:(i + 1) * 12]
                new_test_column[i] = np.nanmean(array_i)
                dummy_test_column[i] = np.sum((~np.isnan(array_i)) * 1)

        data_set_new[test] = new_test_column
        data_set_new['dummy_' + test] = dummy_test_column

    return data_set_new


# def VS_imputation(data_set, vital_signs):
#     N_patients = np.array(data_set.shape[0] / 12).astype(int)

train_features_clean = cleaning(train_features)
test_features_clean = cleaning(test_features)

train_features_clean.to_csv('../data/train_features_clean_mean.csv', header=True, index=False)
test_features_clean.to_csv('../data/test_features_clean_mean.csv', header=True, index=False)
