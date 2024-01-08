import pandas as pd

# read the raw data
data_mat = pd.read_csv("../../data/raw/student-mat.csv")
data_por = pd.read_csv("../../data/raw/student-por.csv")

# assign keys to merge the two dataframes and merge them
keys = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", 
        "Mjob", "Fjob", "activities", "nursery", "internet", "romantic", 
        "famrel", "freetime", "goout", "Dalc", "Walc", "health", "traveltime", 
        "studytime", "guardian", "schoolsup", "famsup", "reason", "higher"]

data = data_mat.merge(data_por, how = "inner", on = keys, suffixes = ("_mat", "_por"))

# assign binary and nominal data for preprocessing
binary_cols_names = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", 
                     "famsup", "activities", "nursery", "higher", "internet", 
                     "romantic", "paid_mat", "paid_por"]

nominal_cols_names = ["Mjob", "Fjob", "reason", "guardian"]

# create one-hot encoders for binary and nominal variables
from sklearn.preprocessing import OneHotEncoder
import numpy as np

binary_encoder = OneHotEncoder(
    sparse_output = False, 
    drop = 'if_binary', 
    dtype = np.int64, 
    feature_name_combiner = lambda input_feature, category : str(input_feature)
)

nominal_encoder = OneHotEncoder(
    sparse_output = False, 
    drop = None, 
    dtype = np.int64,
    feature_name_combiner = lambda input_feature, category : str(input_feature) + "_" + str(category)
)

# apply the encoders
binary_cols = pd.DataFrame(binary_encoder.fit_transform(data[binary_cols_names]))
binary_cols.set_index(data.index)
binary_cols.columns = binary_encoder.get_feature_names_out()

nominal_cols = pd.DataFrame(nominal_encoder.fit_transform(data[nominal_cols_names]))
nominal_cols.set_index(data.index)
nominal_cols.columns = nominal_encoder.get_feature_names_out()

# assign and normalize the numerical variables
numerical_cols_names = ["age", "Medu", "Fedu", "traveltime", "studytime", 
                        "failures_mat", "failures_por", "famrel", "freetime", "goout", 
                        "Dalc", "Walc", "health", "absences_mat", "absences_por", 
                        "G1_mat", "G2_mat", "G3_mat","G1_por", "G2_por", "G3_por"]

numerical_cols = data[numerical_cols_names]
numerical_cols.set_index(data.index)

# create a variable for summed alcohol consumption
numerical_cols["Salc"] = numerical_cols["Dalc"] + numerical_cols["Walc"]

one_to_five_cols = ["Medu", "Fedu", "traveltime", "studytime", "famrel", 
                    "freetime", "goout", "Dalc", "Walc", "health"]

grades = ["G1_mat", "G2_mat", "G3_mat","G1_por", "G2_por", "G3_por"]

failures = ["failures_mat", "failures_por"]

absences = ["absences_mat", "absences_por"]

def normalize(value, min, max):
    return ((value - min) / (max - min))

numerical_cols.loc[:, one_to_five_cols] = numerical_cols[one_to_five_cols].apply(lambda x : normalize(x, 1, 5))
numerical_cols.loc[:, failures] = numerical_cols[failures].apply(lambda x : normalize(x, 0, 3))
numerical_cols.loc[:, grades] = numerical_cols[grades].apply(lambda x : normalize(x, 0, 20))
numerical_cols.loc[:, absences] = numerical_cols[absences].apply(lambda x : normalize(x, 0, 93))
numerical_cols.loc[:, "age"] = numerical_cols["age"].apply(lambda x : normalize(x, 15, 22))
numerical_cols.loc[:, "Salc"] = numerical_cols["Salc"].apply(lambda x : normalize(x, 2, 10))

numerical_cols.round(4)

# concatenate all the variables
frames = [binary_cols, nominal_cols, numerical_cols]
data_processed = pd.concat(frames, axis=1, join="outer", copy=False)

# calculate average grades
data_processed["G1_avg"] = (data_processed["G1_mat"] + data_processed["G1_por"]) / 2
data_processed["G2_avg"] = (data_processed["G2_mat"] + data_processed["G2_por"]) / 2
data_processed["G3_avg"] = (data_processed["G3_mat"] + data_processed["G3_por"]) / 2
data_processed["G_avg"] = (data_processed["G1_avg"] + data_processed["G2_avg"] + data_processed["G3_avg"]) / 3

# save the processed data
data_processed.to_csv("../../data/processed/students_processed.csv")