import numpy as np
import pandas as pd
from progress.bar import IncrementalBar

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

def fill_patient_kNN(array):
    if np.all(np.isnan(array)):
        raise ValueError("Full NaN observations cannot be imputed")
    new_array = np.zeros([12,1])
    for i in range(12):
        if ~np.isnan(array[i]):
            new_array[i] = array[i]
        else:
            new_value = 0
            j=2
            while True:
                offset = int(j/2)* ((-1)**(j%2))
                try:
                    if ~np.isnan(array[i+offset]):
                        new_array[i] = array[i+offset]
                        break
                except IndexError:
                    pass
                finally:
                    j = j+1
    return new_array

def cleaning(data_set):
    # initialize new dataset
    new_columns = patient_characteristics + sum([[test, 'dummy_' + test] for test in tests], []) + vital_signs
    data_set_new = pd.DataFrame((data_set['pid']), columns=new_columns)

    # pid and Age
    N_rows = np.array(data_set.shape[0]).astype(int)
    N_patients = np.array(data_set.shape[0] / 12).astype(int)
    data_set_new[patient_characteristics] = np.array(data_set[patient_characteristics])

    # vital signs
    for VS in vital_signs:
        VS_column = np.array(data_set[VS])
        VS_mean = np.nanmean(VS_column)
        new_VS_column = np.zeros([N_rows, 1])
        for i in range(N_patients):
            if np.all(np.isnan(VS_column[i * 12:(i + 1) * 12])):
                new_VS_column[i * 12:(i + 1) * 12] = VS_mean
            else:
                new_VS_column[i * 12:(i + 1) * 12] = fill_patient_kNN(VS_column[i * 12:(i + 1) * 12])
        data_set_new[VS] = new_VS_column

    # tests
    for test in tests:
        test_column = np.array(data_set[test])
        test_mean = np.nanmean(test_column)
        new_test_column = np.zeros([N_rows, 1])
        dummy_test_column = np.zeros([N_rows, 1])
        for i in range(N_patients):
            if np.all(np.isnan(test_column[i * 12:(i + 1) * 12])):
                new_test_column[i * 12:(i + 1) * 12] = test_mean
                dummy_test_column[i * 12:(i + 1) * 12] = 0
            else:
                new_test_column[i * 12:(i + 1) * 12] = fill_patient_kNN(test_column[i * 12:(i + 1) * 12])
                for j in range(i * 12, (i + 1) * 12):
                    if np.isnan(test_column[j]):
                        dummy_test_column[j] = 0
                    else:
                        dummy_test_column[j] = 1
        data_set_new[test] = new_test_column
        data_set_new['dummy_' + test] = dummy_test_column

    return data_set_new


# def VS_imputation(data_set, vital_signs):
#     N_patients = np.array(data_set.shape[0] / 12).astype(int)

train_features_clean = cleaning(train_features)
test_features_clean = cleaning(test_features)

train_features_clean.to_csv('../data/train_features_clean_all.csv', header=True, index=False)
test_features_clean.to_csv('../data/test_features_clean_all.csv', header=True, index=False)
