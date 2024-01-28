from joblib import load
import pandas as pd

mj = load('model_alcohol_classification.joblib')

data_classification = pd.read_csv('../../data/processed/students_processed.csv', index_col=0)
data_classification_2 = data_classification.head(5)

columns_to_exclude_classification = ['G1_por', 'G2_por', 'G1_mat', 'G2_mat', 'G1_avg', 'G2_avg', 'G3_avg', 'G3_mat', 'G3_por', 'Walc', 'Dalc','Salc']
data_subset_classification = data_classification_2.drop(columns=columns_to_exclude_classification)

#print(mj.n_features_in_,mj.classes_,mj.n_classes_)

#Z jakiegoś powodu ma być 14 inputów i nie moge zrobić .predict()
#mj.predict(data_subset_classification)