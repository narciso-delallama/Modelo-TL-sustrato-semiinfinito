import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Tauc_Lorentz import generar_nyk, ecuaciones_fresnel, calculo_psi_delta

# Configuración de tamaños
label_fontsize = 15
title_fontsize = 20
legend_fontsize = 12
tick_size = 15 

# Cargar datos
df_nn = pd.read_excel('resultados_predicciones_nn.xlsx')
df_pinn = pd.read_excel('resultados_PINN_esp_final.xlsx')

# Parámetros generales
theta_i = 70
energias = np.linspace(0.5, 6.5, 100)
num_espectros = 3

# Filtrar índices válidos (primer delta ≠ 0.0)
valid_indices = []
np.random.seed(42)
shuffled_indices = np.random.permutation(len(df_nn))

for idx in shuffled_indices:
    fila_nn = df_nn.iloc[idx]
    E_real, n_real, k_real, _, _ = generar_nyk(
        fila_nn['A_real'], fila_nn['E0_real'], fila_nn['C_real'],
        fila_nn['Eg_real'], fila_nn['eps_inf_real'], Emin=0.5, Emax=6.5, points=100)
    rp_real, rs_real = ecuaciones_fresnel(n_real, k_real, theta_i)
    _, delta_real = calculo_psi_delta(rp_real, rs_real)
    if np.degrees(delta_real[0]) != 0.0:
        valid_indices.append(idx)
    if len(valid_indices) >= num_espectros:
        break


# Crear figura general
fig, axs = plt.subplots(num_espectros, 2, figsize=(10, 3.5 * num_espectros))
if num_espectros == 1:
    axs = np.expand_dims(axs, axis=0)

for row_idx, idx in enumerate(valid_indices):
    fila_nn = df_nn.iloc[idx]
    fila_pinn = df_pinn.iloc[idx]

    # Real
    E_real, n_real, k_real, _, _ = generar_nyk(
        fila_nn['A_real'], fila_nn['E0_real'], fila_nn['C_real'],
        fila_nn['Eg_real'], fila_nn['eps_inf_real'], Emin=0.5, Emax=6.5, points=100)
    rp_real, rs_real = ecuaciones_fresnel(n_real, k_real, theta_i)
    psi_real, delta_real = calculo_psi_delta(rp_real, rs_real)

    # NN
    E_pred_nn, n_pred_nn, k_pred_nn, _, _ = generar_nyk(
        fila_nn['A_pred'], fila_nn['E0_pred'], fila_nn['C_pred'],
        fila_nn['Eg_pred'], fila_nn['eps_inf_pred'], Emin=0.5, Emax=6.5, points=100)
    rp_pred_nn, rs_pred_nn = ecuaciones_fresnel(n_pred_nn, k_pred_nn, theta_i)
    psi_pred_nn, delta_pred_nn = calculo_psi_delta(rp_pred_nn, rs_pred_nn)

    # PINN
    E_pred_pinn, n_pred_pinn, k_pred_pinn, _, _ = generar_nyk(
        fila_pinn['A_pred'], fila_pinn['E0_pred'], fila_pinn['C_pred'],
        fila_pinn['Eg_pred'], fila_pinn['eps_inf_pred'], Emin=0.5, Emax=6.5, points=100)
    rp_pred_pinn, rs_pred_pinn = ecuaciones_fresnel(n_pred_pinn, k_pred_pinn, theta_i)
    psi_pred_pinn, delta_pred_pinn = calculo_psi_delta(rp_pred_pinn, rs_pred_pinn)

    # Ψ subplot
    axs[row_idx, 0].plot(E_real, np.degrees(psi_real), label='Ψ Real', color='blue')
    axs[row_idx, 0].plot(E_pred_nn, np.degrees(psi_pred_nn), '--', label='Ψ NN', color='orange')
    axs[row_idx, 0].plot(E_pred_pinn, np.degrees(psi_pred_pinn), '--', label='Ψ PINN', color='green')
    axs[row_idx, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[row_idx, 0].set_ylabel('Ψ (°)', fontsize=label_fontsize)
    axs[row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 0].legend(fontsize=legend_fontsize)
    axs[row_idx, 0].grid(True)
    axs[row_idx, 0].tick_params(axis='both', labelsize=legend_fontsize)

    # Δ subplot
    axs[row_idx, 1].plot(E_real, np.degrees(delta_real), label='Δ Real', color='blue')
    axs[row_idx, 1].plot(E_pred_nn, np.degrees(delta_pred_nn), '--', label='Δ NN', color='orange')
    axs[row_idx, 1].plot(E_pred_pinn, np.degrees(delta_pred_pinn), '--', label='Δ PINN', color='green')
    axs[row_idx, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[row_idx, 1].set_ylabel('Δ (°)', fontsize=label_fontsize)
    axs[row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 1].legend(fontsize=legend_fontsize)
    axs[row_idx, 1].grid(True)
    axs[row_idx, 1].tick_params(axis='both', labelsize=legend_fontsize)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#------------------RMSE por Parámetro------------------
axis_label_size = 15      # tamaño etiquetas ejes
title_size = 18           # tamaño títulos
legend_size = 15          # tamaño leyenda
bar_width = 0.35          # ancho de barra
grid_visible = True       # mostrar grid o no
line_width = 2   


# 1. Nombres REALES en el DataFrame
param_grupo1 = ['E0', 'C', 'Eg', 'eps_inf']
param_grupo2 = ['A']

# 2. Etiquetas VISUALES para los ejes
etiquetas_grupo1 = [r'$E_0$/eV', r'$C$/eV', r'$E_g$/eV', r'$\varepsilon_\infty$']
etiquetas_grupo2 = [r'$A$']

# Crear figura general
fig, axs = plt.subplots(num_espectros, 2, figsize=(10, 3.5 * num_espectros))
if num_espectros == 1:
    axs = np.expand_dims(axs, axis=0)

for row_idx, idx in enumerate(valid_indices):
    fila_nn = df_nn.iloc[idx]
    fila_pinn = df_pinn.iloc[idx]

    rmse_nn_grupo1 = []
    rmse_pinn_grupo1 = []
    rmse_nn_grupo2 = []
    rmse_pinn_grupo2 = []

    # Calcular RMSE por parámetro
    for param in param_grupo1:
        rmse_nn = np.sqrt(mean_squared_error([fila_nn[f'{param}_real']], [fila_nn[f'{param}_pred']]))
        rmse_pinn = np.sqrt(mean_squared_error([fila_pinn[f'{param}_real']], [fila_pinn[f'{param}_pred']]))
        rmse_nn_grupo1.append(rmse_nn)
        rmse_pinn_grupo1.append(rmse_pinn)

    for param in param_grupo2:
        rmse_nn = np.sqrt(mean_squared_error([fila_nn[f'{param}_real']], [fila_nn[f'{param}_pred']]))
        rmse_pinn = np.sqrt(mean_squared_error([fila_pinn[f'{param}_real']], [fila_pinn[f'{param}_pred']]))
        rmse_nn_grupo2.append(rmse_nn)
        rmse_pinn_grupo2.append(rmse_pinn)

    x1 = np.arange(len(param_grupo1))
    x2 = np.arange(len(param_grupo2))

    # Subplot grupo 1
    axs[row_idx, 0].bar(x1 - bar_width/2, rmse_nn_grupo1, bar_width, label='NN', color='lightblue', linewidth=line_width)
    axs[row_idx, 0].bar(x1 + bar_width/2, rmse_pinn_grupo1, bar_width, label='PINN', color='lightgreen', linewidth=line_width)
    axs[row_idx, 0].set_xticks(x1)
    axs[row_idx, 0].set_xticklabels(etiquetas_grupo1, fontsize=tick_size)
    axs[row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 0].set_ylabel('RMSE', fontsize=axis_label_size)
    axs[row_idx, 0].legend(fontsize=legend_size)
    axs[row_idx, 0].grid(grid_visible)

    # Subplot grupo 2
    axs[row_idx, 1].bar(x2 - bar_width/2, rmse_nn_grupo2, bar_width, label='NN', color='lightblue', linewidth=line_width)
    axs[row_idx, 1].bar(x2 + bar_width/2, rmse_pinn_grupo2, bar_width, label='PINN', color='lightgreen', linewidth=line_width)
    axs[row_idx, 1].set_xticks(x2)
    axs[row_idx, 1].set_xticklabels(etiquetas_grupo2, fontsize=tick_size)
    axs[row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 1].set_ylabel('RMSE', fontsize=axis_label_size)
    axs[row_idx, 1].legend(fontsize=legend_size)
    axs[row_idx, 1].grid(grid_visible)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
