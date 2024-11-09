"""
Código para análisis y clasificación de datos de préstamos hipotecarios

Este script carga y procesa un conjunto de datos de aprobación de préstamos hipotecarios,
realizando diversas transformaciones y análisis de datos, seguido de la implementación
y evaluación de varios algoritmos de clasificación. Se emplean técnicas de tratamiento
de datos para manejar valores faltantes, desequilibrios en las clases y valores atípicos.
Las siguientes etapas principales se llevan a cabo en el script:

1. **Carga y exploración de datos**:
   - Carga de datos desde un archivo CSV y visualización de la estructura básica del conjunto de datos.
   - Conversión de variables a tipos numéricos, reemplazo de valores faltantes, y verificación
     de valores perdidos.

2. **Transformación de datos**:
   - Relleno de valores faltantes en columnas específicas y tratamiento de valores atípicos.
   - Conversión de ciertas columnas a valores numéricos para facilitar el análisis.

3. **Creación de conjuntos de entrenamiento y prueba**:
   - División del conjunto de datos en conjuntos de entrenamiento y prueba, tanto para el conjunto
     original como para el conjunto con sobremuestreo para balancear las clases.

4. **Entrenamiento y evaluación de modelos**:
   - Entrenamiento de modelos de clasificación usando k-Nearest Neighbors (k-NN), Árbol de Decisión,
     y Naive Bayes, tanto en el conjunto de datos sin balancear como en el balanceado.
   - Evaluación de cada modelo en términos de precisión y matriz de confusión.

5. **Visualización de resultados**:
   - Generación de gráficos de dispersión y boxplots para evaluar la distribución de los datos
     y visualizar los valores atípicos.
   - Visualización de los resultados de los modelos de k-NN en términos de precisión y matriz de confusión.

**Librerías utilizadas**:
- `pandas`, `numpy`: Para manipulación y análisis de datos.
- `scikit-learn`: Para modelos de clasificación y métricas de evaluación.
- `imblearn`: Para técnicas de sobremuestreo (RandomOverSampler).
- `matplotlib`: Para visualización de datos.

**Instrucciones de uso**:
- Asegúrese de que todas las librerías requeridas estén instaladas.
- Los datos deben estar en la ruta especificada como `./SinTratamiento/homeLoanAproval.csv`.

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt

# # Install required packages
# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install scikit-learn

# Load the dataset
datos = pd.read_csv("./SinTratamiento/homeLoanAproval.csv")

# Verify the structure of the data
print(datos.info())

# Summary statistics of the numeric variables
print(datos.describe())

# Perform necessary data transformations
datos["ApplicantIncome"] = pd.to_numeric(datos["ApplicantIncome"], errors="coerce")
datos["CoapplicantIncome"] = pd.to_numeric(datos["CoapplicantIncome"], errors="coerce")
datos["Dependents"] = pd.to_numeric(datos["Dependents"], errors="coerce")
datos["LoanAmount"] = pd.to_numeric(datos["LoanAmount"], errors="coerce")
datos["LoanAmountTerm"] = pd.to_numeric(datos["LoanAmountTerm"], errors="coerce")

# Replace missing values with NA
datos.replace("", np.nan, inplace=True)
datos.replace("3+", 3, inplace=True)

# Print the updated dataset
print(datos.head())

# Check for missing values
print(datos.isnull().sum())

# Drop rows with missing values
datos.dropna(inplace=True)

# Split the dataset into training and testing sets
X = datos[["Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount"]]
y = datos["LoanStatus"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Train and evaluate k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
confusion_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("k-NN Results:")
print("Accuracy:", accuracy_knn)
print("Confusion Matrix:")
print(confusion_matrix_knn)

# Create scatter plot
color_map = {'Y': 'blue', 'N': 'red'}
plt.scatter(X_train["ApplicantIncome"], X_train["LoanAmount"], c=y_train.map(color_map))
plt.title("k-NN without data treatment")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()

# Train and evaluate decision tree classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
confusion_matrix_tree = confusion_matrix(y_test, y_pred_tree)
print("Decision Tree Results:")
print("Accuracy:", accuracy_tree)
print("Confusion Matrix:")
print(confusion_matrix_tree)

# Train and evaluate naive Bayes classifier
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_bayes = naive_bayes.predict(X_test)
accuracy_bayes = accuracy_score(y_test, y_pred_bayes)
confusion_matrix_bayes = confusion_matrix(y_test, y_pred_bayes)
print("Naive Bayes Results:")
print("Accuracy:", accuracy_bayes)
print("Confusion Matrix:")
print(confusion_matrix_bayes)

# Perform data treatment
datos["Dependents"] = datos["Dependents"].fillna(round(datos["Dependents"].mean()))
# datos["Dependents"].fillna(round(datos["Dependents"].mean()), inplace=True)
datos["ApplicantIncome"] = datos["ApplicantIncome"].fillna(datos["ApplicantIncome"].mean())

datos["CoapplicantIncome"] = datos["CoapplicantIncome"].fillna(datos["CoapplicantIncome"].mean())
datos["LoanAmount"] = datos["LoanAmount"].fillna(datos["LoanAmount"].mean())
datos["LoanAmountTerm"] = datos["LoanAmountTerm"].fillna(datos["LoanAmountTerm"].mean())
datos.replace("No", 0)
datos.replace("Yes", 1)
datos["SelfEmployed"] = pd.to_numeric(datos["SelfEmployed"], errors="coerce")
datos["Married"] = pd.to_numeric(datos["Married"], errors="coerce")
datos["SelfEmployed"] = datos["SelfEmployed"].fillna(round(datos["SelfEmployed"].mean()))
datos["Married"] = datos["Married"].fillna(round(datos["Married"].mean()))

# Check for outliers
plt.boxplot([datos["ApplicantIncome"], datos["CoapplicantIncome"], datos["LoanAmount"], datos["LoanAmountTerm"]])
plt.xticks([1, 2, 3, 4], ["Applicant Income", "Coapplicant Income", "Loan Amount", "Loan Amount Term"])
plt.title("Outliers")
plt.show()

# Replace outliers
def replace_outliers(x):
  q1 = np.percentile(x, 25)
  q3 = np.percentile(x, 75)
  iqr = q3 - q1
  x[x < (q1 - 1.5 * iqr)] = q1
  x[x > (q3 + 1.5 * iqr)] = q3
  return x

datos["ApplicantIncome"] = replace_outliers(datos["ApplicantIncome"])
datos["CoapplicantIncome"] = replace_outliers(datos["CoapplicantIncome"])
datos["LoanAmount"] = replace_outliers(datos["LoanAmount"])

# Check class imbalance
print(datos["LoanStatus"].value_counts())
print(datos["LoanStatus"].value_counts(normalize=True))

# Split the dataset into training and testing sets (balanced)
X_balanced = datos[["Dependents", "SelfEmployed", "Married", "ApplicantIncome", "CoapplicantIncome", "LoanAmount"]]
y_balanced = datos["LoanStatus"]
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=123)

# Perform oversampling
ros = RandomOverSampler(random_state=123)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train_balanced, y_train_balanced)

# Train and evaluate k-NN classifier (balanced)
knn_balanced = KNeighborsClassifier(n_neighbors=3)
knn_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_bal_knn = knn_balanced.predict(X_test_balanced)
accuracy_bal_knn = accuracy_score(y_test_balanced, y_pred_bal_knn)
confusion_matrix_bal_knn = confusion_matrix(y_test_balanced, y_pred_bal_knn)
print("k-NN Results (balanced):")
print("Accuracy:", accuracy_bal_knn)
print("Confusion Matrix:")
print(confusion_matrix_bal_knn)

# Create scatter plot (balanced)
color_map = {'Y': 'blue', 'N': 'red'}
plt.scatter(X_train_balanced["ApplicantIncome"], X_train_balanced["LoanAmount"], c=y_train_balanced.map(color_map))
plt.title("k-NN with data treatment (balanced)")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()

# Train and evaluate decision tree classifier (balanced)
tree_balanced = DecisionTreeClassifier()
tree_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_bal_tree = tree_balanced.predict(X_test_balanced)
accuracy_bal_tree = accuracy_score(y_test_balanced, y_pred_bal_tree)
confusion_matrix_bal_tree = confusion_matrix(y_test_balanced, y_pred_bal_tree)
print("Decision Tree Results (balanced):")
print("Accuracy:", accuracy_bal_tree)
print("Confusion Matrix:")
print(confusion_matrix_bal_tree)

# Train and evaluate naive Bayes classifier (balanced)
naive_bayes_balanced = GaussianNB()
naive_bayes_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_bal_bayes = naive_bayes_balanced.predict(X_test_balanced)
accuracy_bal_bayes = accuracy_score(y_test_balanced, y_pred_bal_bayes)
confusion_matrix_bal_bayes = confusion_matrix(y_test_balanced, y_pred_bal_bayes)
print("Naive Bayes Results (balanced):")
print("Accuracy:", accuracy_bal_bayes)
print("Confusion Matrix:")
print(confusion_matrix_bal_bayes)