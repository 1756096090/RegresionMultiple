import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error  # ← Importar métricas faltantes

# Cargar datos
dataSet = pd.read_csv('Advertising (1).csv')
X = dataSet[['TV', 'Radio']].values
y = dataSet['Sales'].values

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Entrenar el modelo
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicciones
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Gráfico de entrenamiento
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train, color='red', label='Datos de entrenamiento')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='blue', lw=2, label='Línea ideal')
plt.title('Ventas Reales vs. Ventas Predichas (Entrenamiento)')
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de prueba
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='green', label='Datos de prueba')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', lw=2, label='Línea ideal')
plt.title('Ventas Reales vs. Ventas Predichas (Prueba)')
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.legend()
plt.grid(True)
plt.show()

# Modelo statsmodels con test
X_test_sm = sm.add_constant(X_test)
model = sm.OLS(y_test, X_test_sm).fit()
print(model.summary())

# Métricas solo del test
r_value, _ = pearsonr(y_test, y_pred_test)
print(f"R (correlación): {r_value:.4f}")

r2 = r2_score(y_test, y_pred_test)
print(f"R² (coeficiente de determinación): {r2:.4f}")

mae = mean_absolute_error(y_test, y_pred_test)
print(f"MAE (Error Absoluto Medio): {mae:.4f}")
