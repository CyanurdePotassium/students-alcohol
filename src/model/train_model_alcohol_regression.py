import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.tree import plot_tree

# python3 -m pip install -U git+https://github.com/ray-project/tune-sklearn.git && pip install 'ray[tune]'
from tune_sklearn import TuneSearchCV
from sklearn import metrics

### PRZYGOTOWANIE DANYCH ###
############################

data_regression = pd.read_csv('../../data/processed/students_processed.csv', index_col=0)

# Lista kolumn do wykluczenia
columns_to_exclude_regression = ['G1_por', 'G2_por', 'G1_mat', 'G2_mat', 'G1_avg', 'G2_avg', 'G3_avg', 'G3_mat', 'G3_por', 'Walc', 'Dalc']
data_subset_regression = data_regression.drop(columns=columns_to_exclude_regression)

# Podział na zbiór cech i zbiór etykiet
X_regression = data_subset_regression.drop('Salc', axis=1)
y_regression = data_subset_regression['Salc']  # Zmienna celu dla regresji

# Podział zbioru danych na zestaw treningowy i testowy
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

### WYZNACZANIE HIPERPARAMETRÓW ###
###################################

# Utworzenie instancji RandomForestregressor
random_forest_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

# Hiperparametry do dostosowania
regression_hyperparameters = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
}

# Wybór najlepszych hiperparametrów za pomocą walidacji krzyżowej

# wersja zółwia
#regression_grid_search = GridSearchCV(random_forest_regressor, regression_hyperparameters, cv=5)

regression_grid_search = TuneSearchCV(estimator=random_forest_regressor,
                               param_distributions=regression_hyperparameters,
                               n_trials=20,
                               n_jobs=8,
                               cv=5)
regression_grid_search.fit(X_train_regression, y_train_regression)

### RYSOWANIE DRZEWA DECYZYJNEGO ###
####################################

# Zdobądź najlepsze estymatory
best_rf_model_regression = regression_grid_search.best_estimator_

# Indeks drzewa do wizualizacji
tree_index_to_plot_regression = 0

# Narysuj drzewo
plt.figure(figsize=(20, 10))
plot_tree(best_rf_model_regression.estimators_[tree_index_to_plot_regression], filled=True, feature_names=X_train_regression.columns)
plt.title(f'Przykładowe drzewo decyzyjne {tree_index_to_plot_regression}')
plt.show()

### WYKRES HIPERPARAMETRÓW ###
##############################

# Zdobądź wyniki walidacji krzyżowej
results_regression = regression_grid_search.cv_results_
params_regression = results_regression['params']
mean_test_scores_regression = results_regression['mean_test_score']
best_index_regression = np.argmax(mean_test_scores_regression)

# Skonwertuj do DataFrame
hyperparameters_df_regression = pd.DataFrame(params_regression)
hyperparameters_df_regression['mean_test_score'] = mean_test_scores_regression

# Narysuj wykres
fig_regression = px.parallel_coordinates(
    hyperparameters_df_regression,
    color='mean_test_score',
    color_continuous_scale='Viridis',
    labels={'color': 'Mean Test Score'},
    dimensions=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
)
best_index_regression = hyperparameters_df_regression['mean_test_score'].idxmax()
fig_regression.add_annotation(xref='paper', yref='paper', x=0, y=0,
                   text=f'Najlepszy <br>Mean test score: {hyperparameters_df_regression["mean_test_score"].iloc[best_index_regression]:.3f}',
                   showarrow=False, font=dict(color='red', size=14))
fig_regression.show()

# Wydrukowanie najlepszych hiperparametrów
best_params_regression = regression_grid_search.best_params_

### TRENOWANIE MODELU ###
#########################

# Trenowanie lasu losowego na całym zestawie treningowym z najlepszymi hiperparametrami
random_forest_regressor.set_params(**best_params_regression)
random_forest_regressor.fit(X_train_regression, y_train_regression)

# Selekcja cech
sfm_regression = SelectFromModel(random_forest_regressor, threshold='mean')
sfm_regression.fit(X_train_regression, y_train_regression)

# Zdobądź indeksy i nazwy wybranych cech
selected_feature_indices_regression = sfm_regression.get_support(indices=True)
selected_feature_names_regression = X_train_regression.columns[selected_feature_indices_regression]

### WYKRES WAŻNOŚCI CECH ###
############################

# Narysuj wykres ważności cech
feature_importances_regression = pd.Series(random_forest_regressor.feature_importances_, index=X_train_regression.columns)
sorted_feature_importances_regression = feature_importances_regression[selected_feature_names_regression].sort_values().to_frame(name='Istotność')

plt.figure(figsize=(10, 6))
sns.barplot(x='Istotność', y=sorted_feature_importances_regression.index, data=sorted_feature_importances_regression, palette='viridis')
plt.title('Istotność cech')
plt.xlabel('Istotność')
plt.ylabel('Cecha')
plt.show()

# Stwórz dataset tylko z najważniejszymi cechami
X_train_selected_regression = sfm_regression.transform(X_train_regression)
X_test_selected_regression = sfm_regression.transform(X_test_regression)

# Przetrenuj model na danych z wybranymi cechami
random_forest_regressor.fit(X_train_selected_regression, y_train_regression)

# Przewidzenie wartości dla danych testowych
y_pred_selected_regression = random_forest_regressor.predict(X_test_selected_regression)

### EWAULACJA MODELU ###
########################

accuracy_selected_regression = random_forest_regressor.score(X_test_selected_regression, y_test_regression)
mse_selected_regression = mean_squared_error(y_test_regression, y_pred_selected_regression)

print(f'Dokładność modelu dla regresji: {accuracy_selected_regression}')
print(f'Błąd średniokwadratowy modelu dla regresji: {mse_selected_regression}')
print(f'Najlepsze hiperparametry dla regresji: {best_params_regression}')
