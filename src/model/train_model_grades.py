import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.tree import plot_tree

### PRZYGOTOWANIE DANYCH ###
############################
data = pd.read_csv('../../data/processed/students_processed.csv', index_col=0)

# lista kolumn do wykluczenia
# G1-3 - oceny i ich średnie, w tym modelu ich nie potrzebujemy
# Salc - suma Walc i Dalc
columns_to_exclude = columns_to_exclude = ['G1_por', 'G2_por', 'G1_mat', 'G2_mat', 'G1_avg', 'G2_avg', 'G3_avg', 'G3_mat', 'G3_por', 'Salc']
data_subset = data.drop(columns=columns_to_exclude)

# Podział na zbiór cech i zbiór etykiet
X = data_subset.drop('G_avg', axis=1)
y = (data['G_avg'] >= 0.5).astype(int)  # Klasyfikacja binarna, próg "zaliczenia" 50%

# Podział zbioru danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### WYZNACZANIE HIPERPARAMETRÓW ###
###################################

# Utworzenie instancji RandomForestRegressor
random_forest_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

# Hiperparametry do dostosowania
hyperparameters = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
}

# Wybór najlepszych hiperparametrów za pomocą walidacji krzyżowej
grid_search = GridSearchCV(random_forest_classifier, hyperparameters, cv=5)
grid_search.fit(X_train, y_train)

### RYSOWANIE DRZEWA DECYZYJNEGO ###
####################################

# Zdobądź najlepsze estymatory
best_rf_model = grid_search.best_estimator_

# Indeks drzewa do wizualizacji
tree_index_to_plot = 0

# Narysuj drzewo
plt.figure(figsize=(20, 10))
plot_tree(best_rf_model.estimators_[tree_index_to_plot], filled=True, feature_names=X_train.columns)
plt.title(f'Przykładowe drzewo decyzyjne {tree_index_to_plot}')
plt.show()

### WYKRES HIPERPARAMETRÓW ###
##############################

# Zdobądź wyniki walidacji krzyżowej
results = grid_search.cv_results_
params = results['params']
mean_test_scores = results['mean_test_score']
best_index = np.argmax(mean_test_scores)

# Skonwertuj do DataFrame
hyperparameters_df = pd.DataFrame(params)
hyperparameters_df['mean_test_score'] = mean_test_scores

# Narysuj wykres
fig = px.parallel_coordinates(
    hyperparameters_df,
    color='mean_test_score',
    color_continuous_scale='Viridis',
    labels={'color': 'Mean Test Score'},
    dimensions=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
)
best_index = hyperparameters_df['mean_test_score'].idxmax()
fig.add_annotation(xref='paper', yref='paper', x=0, y=0,
                   text=f'Najlepszy <br>Mean test score: {hyperparameters_df["mean_test_score"].iloc[best_index]:.3f}',
                   showarrow=False, font=dict(color='red', size=14))
fig.show()

# Wydrukowanie najlepszych hiperparametrów
best_params = grid_search.best_params_

### TRENOWANIE MODELU ###
#########################

# Trenowanie lasu losowego na całym zestawie treningowym z najlepszymi hiperparametrami
random_forest_classifier.set_params(**best_params)
random_forest_classifier.fit(X_train, y_train)

# Selekcja cech
sfm = SelectFromModel(random_forest_classifier, threshold='mean')
sfm.fit(X_train, y_train)

# Zdobądź indeksy i nazwy wybranych cech
selected_feature_indices = sfm.get_support(indices=True)
selected_feature_names = X_train.columns[selected_feature_indices]

### WYKRES WAŻNOŚCI CECH ###
############################

# Narysuj wykres ważności cech
feature_importances = pd.Series(random_forest_classifier.feature_importances_, index=X_train.columns)
sorted_feature_importances = feature_importances[selected_feature_names].sort_values().to_frame(name='Istotność')

plt.figure(figsize=(10, 6))
sns.barplot(x='Istotność', y=sorted_feature_importances.index, data=sorted_feature_importances, palette='viridis')
plt.title('Istotność cech')
plt.xlabel('Istotność')
plt.ylabel('Cecha')
plt.show()

# Stwórz dataset tylko z najwaniejszymi cechami
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Przetrenuj model na danych z wybranymi cechami
random_forest_classifier.fit(X_train_selected, y_train)

# Przewidzenie wartości dla danych testowych
y_pred_selected = random_forest_classifier.predict(X_test_selected)

### EWAULACJA MODELU ###
########################

accuracy_selected = accuracy_score(y_test, y_pred_selected)
mse_selected = mean_squared_error(y_test, y_pred_selected)

print(f'Dokładność modelu: {accuracy_selected}')
print(f'Błąd średniokwadratowy modelu: {mse_selected}')
print(f'Najlepsze hiperparametry:" {best_params}')