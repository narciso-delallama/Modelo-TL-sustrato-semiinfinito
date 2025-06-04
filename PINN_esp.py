import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from scipy.interpolate import interp1d
from Tauc_Lorentz import generar_nyk, ecuaciones_fresnel, calculo_psi_delta

# Cargar datos
df = pd.read_csv('dataset_elipsometria_10000.csv')
num_puntos_interp = 50
energias_interp = np.linspace(df['E'].min(), df['E'].max(), num_puntos_interp)

# Preprocesamiento
grupos = df.groupby(['A', 'E0', 'C', 'Eg', 'eps_inf'])
X, y = [], []

for (A, E0, C, Eg, eps_inf), group in grupos:
    group_sorted = group.sort_values('E')
    if group_sorted['E'].nunique() < 2:
        continue
    interp_psi = interp1d(group_sorted['E'], group_sorted['psi_deg'], kind='linear', fill_value="extrapolate")
    interp_delta = interp1d(group_sorted['E'], group_sorted['delta_deg'], kind='linear', fill_value="extrapolate")
    espectro = np.concatenate([interp_psi(energias_interp), interp_delta(energias_interp)])
    X.append(espectro)
    y.append([A, C, Eg, E0, eps_inf])

X = np.array(X)
y = np.array(y)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

etiquetas_estrato = pd.qcut(y[:, 3], q=10, labels=False)
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Constantes físicas
theta_i = 70  # ángulo de incidencia
lambda_fisica = 1.0  # peso del término físico en la loss

def loss_pinn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Split de y_true en: [params verdaderos, espectro real]
    y_true_params = y_true[:, :5]
    y_true_spectro = y_true[:, 5:]

    # Desnormaliza predicción
    y_pred_descaled = y_pred * tf.constant(scaler_y.scale_, dtype=tf.float32) + tf.constant(scaler_y.mean_, dtype=tf.float32)
    A, C, Eg, E0, eps_inf = [y_pred_descaled[:, i] for i in range(5)]

    def calculate_psi_delta(A, C, Eg, E0, eps_inf):
        psi_list, delta_list = [], []
        for i in range(len(A)):
            _, n, k, _, _ = generar_nyk(A[i], E0[i], C[i], Eg[i], eps_inf[i], Emin=0.5, Emax=6.5, points=num_puntos_interp)
            rp, rs = ecuaciones_fresnel(n, k, theta_i)
            psi, delta = calculo_psi_delta(rp, rs)
            psi_list.append(np.degrees(psi))
            delta_list.append(np.degrees(delta))
        return np.array(psi_list, dtype=np.float32), np.array(delta_list, dtype=np.float32)


    psi_pred, delta_pred = tf.numpy_function(
        func=calculate_psi_delta,
        inp=[A, C, Eg, E0, eps_inf],
        Tout=[tf.float32, tf.float32]
    )

    psi_delta_pred = tf.concat([psi_pred, delta_pred], axis=1)

    # Calcula pérdidas
    loss_parametros = tf.reduce_mean(tf.square(y_true_params - y_pred))
    loss_fisica = tf.reduce_mean(tf.square(psi_delta_pred - y_true_spectro))

    return loss_parametros + lambda_fisica * loss_fisica




# Entrenamiento con PINN
all_y_true, all_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, etiquetas_estrato)):
    print(f"\nEntrenando fold {fold + 1}/{n_splits}")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train = np.concatenate([y_scaled[train_idx], X[train_idx]], axis=1)
    y_test = np.concatenate([y_scaled[test_idx], X[test_idx]], axis=1)


    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(5)
    ])

    model.compile(optimizer='adam', loss=loss_pinn)
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    y_pred_scaled = model.predict(X_test)
    # Separar parámetros predichos y reales (solo los primeros 5)
    y_test_params_scaled = y_test[:, :5]
    y_pred_params_scaled = model.predict(X_test)

    # Desnormalizar solo los parámetros
    y_true = scaler_y.inverse_transform(y_test_params_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_params_scaled)

    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

# Evaluación
y_true_full = np.vstack(all_y_true)
y_pred_full = np.vstack(all_y_pred)
mae_final = mean_absolute_error(y_true_full, y_pred_full, multioutput='raw_values')
print("\nMAE final por parámetro:", mae_final)
model.save('modelo_PINN_elipsometria.h5')

# Nombres de los parámetros
param_names = ['A', 'C', 'Eg', 'E0', 'eps_inf']

# Crear diccionario con columnas intercaladas
data = {}
for i, name in enumerate(param_names):
    data[f'{name}_real'] = y_true_full[:, i]
    data[f'{name}_pred'] = y_pred_full[:, i]

# Crear DataFrame
df_resultados = pd.DataFrame(data)

# Guardar en Excel (opcional)
df_resultados.to_excel('resultados_predicciones_PINN.xlsx', index=False)
print("Archivo guardado como 'resultados_predicciones_PINN.xlsx'")

# Selección aleatoria de 3 ejemplos
indices_aleatorios = random.sample(range(len(df_resultados)), 3)

# Energías de simulación
energias_interp = np.linspace(0.5, 6.5, 100)

# Gráfica
fig, axs = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Comparación de Ψ y Δ reales vs predichos (PINN)')

for fila_idx, i in enumerate(indices_aleatorios):
    fila = df_resultados.iloc[i]

    # PARÁMETROS REALES
    A_real = fila['A_real']
    C_real = fila['C_real']
    Eg_real = fila['Eg_real']
    E0_real = fila['E0_real']
    eps_inf_real = fila['eps_inf_real']
    
    E_real, n_real, k_real, _, _ = generar_nyk(A_real, E0_real, C_real, Eg_real, eps_inf_real, 
                                               Emin=energias_interp.min(), Emax=energias_interp.max(), 
                                               points=len(energias_interp))
    rp_real, rs_real = ecuaciones_fresnel(n_real, k_real, theta_i)
    psi_real, delta_real = calculo_psi_delta(rp_real, rs_real)

    # PARÁMETROS PREDICHOS
    A_pred = fila['A_pred']
    C_pred = fila['C_pred']
    Eg_pred = fila['Eg_pred']
    E0_pred = fila['E0_pred']
    eps_inf_pred = fila['eps_inf_pred']
    
    E_pred, n_pred, k_pred, _, _ = generar_nyk(A_pred, E0_pred, C_pred, Eg_pred, eps_inf_pred, 
                                               Emin=energias_interp.min(), Emax=energias_interp.max(), 
                                               points=len(energias_interp))
    rp_pred, rs_pred = ecuaciones_fresnel(n_pred, k_pred, theta_i)
    psi_pred, delta_pred = calculo_psi_delta(rp_pred, rs_pred)

    # GRAFICAR Ψ
    axs[fila_idx, 0].plot(E_real, np.degrees(psi_real), label='Ψ real', linestyle='--')
    axs[fila_idx, 0].plot(E_pred, np.degrees(psi_pred), label='Ψ predicho')
    axs[fila_idx, 0].set_ylabel('Ψ (°)')
    axs[fila_idx, 0].legend()
    axs[fila_idx, 0].set_title(f'Conjunto {fila_idx+1} - Ψ')

    # GRAFICAR Δ
    axs[fila_idx, 1].plot(E_real, np.degrees(delta_real), label='Δ real', linestyle='--')
    axs[fila_idx, 1].plot(E_pred, np.degrees(delta_pred), label='Δ predicho')
    axs[fila_idx, 1].set_ylabel('Δ (°)')
    axs[fila_idx, 1].legend()
    axs[fila_idx, 1].set_title(f'Conjunto {fila_idx+1} - Δ')

# Etiquetas finales
for ax in axs[-1]:
    ax.set_xlabel('Energía (eV)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
