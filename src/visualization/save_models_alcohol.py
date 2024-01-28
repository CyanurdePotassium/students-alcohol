from joblib import dump
import sys

sys.path.append('..')

# import modelu alkohol classification
from model.train_model_alcohol_classification import random_forest_classifier

# Zapis modelu
nazwa_pliku = "model_alcohol_classification.joblib"
dump(random_forest_classifier, nazwa_pliku)

# import modelu alkohol regression
from model.train_model_alcohol_regression import random_forest_regressor

# Zapis modelu
nazwa_pliku = "model_alcohol_regression.joblib"
dump(random_forest_regressor, nazwa_pliku)
