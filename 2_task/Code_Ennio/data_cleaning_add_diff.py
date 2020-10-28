import numpy as np
import pandas as pd

def add_diff(train_path, test_path):
    #example:
    #train_path = "../data/train_features_clean_columned.csv"
    #test_path = "../data/test_features_clean_columned.csv"

    # ---------------------------------------------------------
    # ----------------- DATA IMPORT ------------------------
    # ---------------------------------------------------------
    train_features_all = pd.read_csv("../data/train_features_clean_all.csv")
    test_features_all = pd.read_csv("../data/test_features_clean_all.csv")
    train_features = pd.read_csv(train_path)
    test_features = pd.read_csv(test_path)

    patient_characteristics = ["pid", "Age"]  # TIME VARIABLE IS EXCLUDED
    vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate", 'Temp']
    tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess',
             'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
             'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
             'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
             'Bilirubin_total', 'TroponinI', 'pH']


    # ---------------------------------------------------------
    # ----------------- DATA AUGMENTATION ------------------------
    # ---------------------------------------------------------

    def fill_diff_features(dataset, dataset_all):
        dataset_VS = dataset_all[vital_signs]

        # calculate number of extrema
        def number_of_extrema(array):
            diff = np.diff(array)
            diff = (diff >= 0) * 1 + (diff < 0) * -1
            extrema = diff[:-1] * diff[1:]
            return extrema

        extrema = np.apply_along_axis(number_of_extrema, 0, dataset_VS)
        extrema = 1 * (extrema[
                       np.multiply((np.arange(1, extrema.shape[0] + 1) % 12) > 0,
                                   (np.arange(2, extrema.shape[0] + 2) % 12) > 0),
                       :] == -1)
        sum_every_ten_elements = lambda array: np.sum(array.reshape(int(array.shape[0] / 10), 10), axis=1)
        N_extrema = np.apply_along_axis(sum_every_ten_elements, 0, extrema)

        # calculate number of differences
        dataset_diff = np.diff(dataset_VS, axis=0)
        dataset_diff = dataset_diff[(np.arange(1, dataset_diff.shape[0] + 1) % 12) > 0, :]
        mean_every_eleven_elements = lambda array: np.mean(array.reshape(int(array.shape[0] / 11), 11), axis=1)
        median_every_eleven_elements = lambda array: np.median(array.reshape(int(array.shape[0] / 11), 11), axis=1)
        max_every_eleven_elements = lambda array: np.min(array.reshape(int(array.shape[0] / 11), 11), axis=1)
        min_every_eleven_elements = lambda array: np.min(array.reshape(int(array.shape[0] / 11), 11), axis=1)
        diff_mean = np.apply_along_axis(mean_every_eleven_elements, 0, dataset_diff)
        diff_median = np.apply_along_axis(median_every_eleven_elements, 0, dataset_diff)
        diff_max = np.apply_along_axis(max_every_eleven_elements, 0, dataset_diff)
        diff_min = np.apply_along_axis(min_every_eleven_elements, 0, dataset_diff)

        print()

        diff_features_suffixes = ['_n_extrema', '_diff_mean', '_diff_median', '_diff_max', '_diff_min']
        diff_features = sum(
            [[VS + diff_features_suffix for VS in vital_signs] for diff_features_suffix in diff_features_suffixes], [])

        dataset_diff = pd.DataFrame(columns=diff_features,
                                    index=dataset.index,
                                    data=np.column_stack((N_extrema, diff_mean, diff_median, diff_max, diff_min)))

        dataset_features_diff = pd.concat([dataset, dataset_diff], axis=1, sort=False)
        return dataset_features_diff


    train_features_diff = fill_diff_features(train_features, train_features_all)
    test_features_diff = fill_diff_features(test_features, test_features_all)

    train_features_diff.to_csv(train_path[:-4] + '_diff.csv', header=True, index=False)
    test_features_diff.to_csv(test_path[:-4] + '_diff.csv', header=True, index=False)
