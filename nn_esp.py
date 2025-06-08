import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import random
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from scipy.interpolate import interp1d
from Tauc_Lorentz import generar_nyk, ecuaciones_fresnel, calculo_psi_delta
import matplotlib.pyplot as plt

# Cargar el dataset generado
df = pd.read_csv('dataset_elipsometria_10000.csv')

# Parámetros del espectro a interpolar
num_puntos_interp = 50
energias_interp = np.linspace(df['E'].min(), df['E'].max(), num_puntos_interp)

# Agrupar por espectro completo (cada conjunto de parámetros)
grupos = df.groupby(['A', 'E0', 'C', 'Eg', 'eps_inf'])
X = []
y = []

for (A, E0, C, Eg, eps_inf), group in grupos:
    group_sorted = group.sort_values('E')
    if group_sorted['E'].nunique() < 2:
        continue  # No se puede interpolar con menos de 2 puntos
    interp_psi = interp1d(group_sorted['E'], group_sorted['psi_deg'], kind='linear', fill_value="extrapolate")
    interp_delta = interp1d(group_sorted['E'], group_sorted['delta_deg'], kind='linear', fill_value="extrapolate")
    espectro = np.concatenate([interp_psi(energias_interp), interp_delta(energias_interp)])
    X.append(espectro)
    y.append([A, C, Eg, E0, eps_inf])

X = np.array(X)
y = np.array(y)

# Normalización
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Crear etiquetas discretas para StratifiedKFold (basado en binning de E0)
etiquetas_estrato = pd.qcut(y[:, 3], q=10, labels=False)

# Stratified K-Fold
n_splits = 10
skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, etiquetas_estrato)):
    print(f"\nEntrenando fold {fold + 1}/{n_splits}")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

    # Modelo
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(5)  # A, C, Eg, E0, eps_inf
    ])

    model.compile(optimizer='adam', loss='mse')
    # Callback de EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entrenamiento
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluación
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

# Comparativa final
y_true_full = np.vstack(all_y_true)
y_pred_full = np.vstack(all_y_pred)


mae_final = mean_absolute_error(y_true_full, y_pred_full, multioutput='raw_values')
print("\nMAE final por parámetro:", mae_final)
rmse_final = np.sqrt(mean_squared_error(y_true_full, y_pred_full, multioutput='raw_values'))
print("\nRMSE final por parámetro:", rmse_final)

# Guardar modelo final
model.save('modelo_elipsometria_10000.h5')
print("Modelo guardado como 'modelo_elipsometria_10000.h5'")


theta_i = 70


# Crear DataFrame con columnas intercaladas real/predicho
columnas = ['A', 'C', 'Eg', 'E0', 'eps_inf']
data = {}

for i, nombre in enumerate(columnas):
    data[f'{nombre}_real'] = y_true_full[:, i]
    data[f'{nombre}_pred'] = y_pred_full[:, i]

df_resultados = pd.DataFrame(data)

# Guardar como archivo Excel
df_resultados.to_excel('resultados_predicciones_nn.xlsx', index=False)
print("Archivo Excel guardado como 'resultados_predicciones_nn.xlsx'")

