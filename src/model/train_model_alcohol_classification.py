import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.tree import plot_tree

#python3 -m pip install -U git+https://github.com/ray-project/tune-sklearn.git && pip install 'ray[tune]'
from tune_sklearn import TuneSearchCV

### PRZYGOTOWANIE DANYCH ###
############################

data_classification = pd.read_csv('../../data/processed/students_processed.csv', index_col=0)

# Lista kolumn do wykluczenia
columns_to_exclude_classification = ['G1_por', 'G2_por', 'G1_mat', 'G2_mat', 'G1_avg', 'G2_avg', 'G3_avg', 'G3_mat', 'G3_por', 'Walc', 'Dalc']
data_subset_classification = data_classification.drop(columns=columns_to_exclude_classification)

# Podział na zbiór cech i zbiór etykiet
X_classification = data_subset_classification.drop('Salc', axis=1)
y_classification = (data_classification['Salc'] >= 0.4).astype(int)  # Klasyfikacja binarna, próg "spozycia" 40%

# Podział zbioru danych na zestaw treningowy i testowy
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

### WYZNACZANIE HIPERPARAMETRÓW ###
###################################

# Utworzenie instancji RandomForestClassifier
random_forest_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

# Hiperparametry do dostosowania
classification_hyperparameters = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
}

# Wybór najlepszych hiperparametrów za pomocą walidacji krzyżowej

# wersja zółwia
#classification_grid_search = GridSearchCV(random_forest_classifier, classification_hyperparameters, cv=5)

classification_grid_search = TuneSearchCV(estimator=random_forest_classifier,
                               param_distributions=classification_hyperparameters,
                               n_trials=20,
                               n_jobs=8,
                               cv=5)

classification_grid_search.fit(X_train_classification, y_train_classification)

### RYSOWANIE DRZEWA DECYZYJNEGO ###
####################################

# Zdobądź najlepsze estymatory
best_rf_model_classification = classification_grid_search.best_estimator_

# Indeks drzewa do wizualizacji
tree_index_to_plot_classification = 0

# Narysuj drzewo
plt.figure(figsize=(20, 10))
plot_tree(best_rf_model_classification.estimators_[tree_index_to_plot_classification], filled=True, feature_names=X_train_classification.columns)
plt.title(f'Przykładowe drzewo decyzyjne {tree_index_to_plot_classification}')
plt.show()

### WYKRES HIPERPARAMETRÓW ###
##############################

# Zdobądź wyniki walidacji krzyżowej
results_classification = classification_grid_search.cv_results_
params_classification = results_classification['params']
mean_test_scores_classification = results_classification['mean_test_score']
best_index_classification = np.argmax(mean_test_scores_classification)

# Skonwertuj do DataFrame
hyperparameters_df_classification = pd.DataFrame(params_classification)
hyperparameters_df_classification['mean_test_score'] = mean_test_scores_classification

# Narysuj wykres
fig_classification = px.parallel_coordinates(
    hyperparameters_df_classification,
    color='mean_test_score',
    color_continuous_scale='Viridis',
    labels={'color': 'Mean Test Score'},
    dimensions=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
)
best_index_classification = hyperparameters_df_classification['mean_test_score'].idxmax()
fig_classification.add_annotation(xref='paper', yref='paper', x=0, y=0,
                   text=f'Najlepszy <br>Mean test score: {hyperparameters_df_classification["mean_test_score"].iloc[best_index_classification]:.3f}',
                   showarrow=False, font=dict(color='red', size=14))
fig_classification.show()

# Wydrukowanie najlepszych hiperparametrów
best_params_classification = classification_grid_search.best_params_

### TRENOWANIE MODELU ###
#########################

# Trenowanie lasu losowego na całym zestawie treningowym z najlepszymi hiperparametrami
random_forest_classifier.set_params(**best_params_classification)
random_forest_classifier.fit(X_train_classification, y_train_classification)

# Selekcja cech
sfm_classification = SelectFromModel(random_forest_classifier, threshold='mean')
sfm_classification.fit(X_train_classification, y_train_classification)

# Zdobądź indeksy i nazwy wybranych cech
selected_feature_indices_classification = sfm_classification.get_support(indices=True)
selected_feature_names_classification = X_train_classification.columns[selected_feature_indices_classification]

### WYKRES WAŻNOŚCI CECH ###
############################

# Narysuj wykres ważności cech
feature_importances_classification = pd.Series(random_forest_classifier.feature_importances_, index=X_train_classification.columns)
sorted_feature_importances_classification = feature_importances_classification[selected_feature_names_classification].sort_values().to_frame(name='Istotność')

plt.figure(figsize=(10, 6))
sns.barplot(x='Istotność', y=sorted_feature_importances_classification.index, data=sorted_feature_importances_classification, palette='viridis')
plt.title('Istotność cech')
plt.xlabel('Istotność')
plt.ylabel('Cecha')
plt.show()

# Stwórz dataset tylko z najważniejszymi cechami
X_train_selected_classification = sfm_classification.transform(X_train_classification)
X_test_selected_classification = sfm_classification.transform(X_test_classification)

# Przetrenuj model na danych z wybranymi cechami
random_forest_classifier.fit(X_train_selected_classification, y_train_classification)

# Przewidzenie wartości dla danych testowych
y_pred_selected_classification = random_forest_classifier.predict(X_test_selected_classification)

### EWAULACJA MODELU ###
########################

accuracy_selected_classification = accuracy_score(y_test_classification, y_pred_selected_classification)
mse_selected_classification = mean_squared_error(y_test_classification, y_pred_selected_classification)

print(f'Dokładność modelu dla klasyfikacji: {accuracy_selected_classification}')
print(f'Błąd średniokwadratowy modelu dla klasyfikacji: {mse_selected_classification}')
print(f'Najlepsze hiperparametry dla klasyfikacji: {best_params_classification}')