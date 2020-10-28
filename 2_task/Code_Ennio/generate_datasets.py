from Code_Ennio.data_cleaning_add_diff import add_diff
exec(open("data_cleaning_mean.py").read())
exec(open("data_cleaning_all.py").read())
exec(open("data_reshape.py").read())
add_diff("../data/train_features_clean_columned.csv", "../data/test_features_clean_columned.csv")
add_diff("../data/train_features_clean_mean.csv", "../data/test_features_clean_mean.csv")

