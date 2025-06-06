import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Tauc_Lorentz import generar_nyk, ecuaciones_fresnel, calculo_psi_delta


# Cargar datos
df_nn = pd.read_excel('resultados_predicciones_nn.xlsx')
df_pinn = pd.read_excel('resultados_PINN_esp_final.xlsx')

# Parámetros generales
theta_i = 70
energias = np.linspace(0.5, 6.5, 100)
num_indices = 3

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
    if len(valid_indices) >= num_indices:
        break


# --- Parámetros de tamaño de texto ---
label_fontsize = 20    # tamaño etiquetas ejes
title_fontsize = 22    # tamaño título
legend_fontsize = 15   # tamaño leyenda
tick_size = 20         # tamaño de los números en ejes


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Tauc_Lorentz import generar_nyk, ecuaciones_fresnel, calculo_psi_delta

# Configuración de tamaños
label_fontsize = 20
legend_fontsize = 15
tick_size = 20

# Cargar datos
df_nn = pd.read_excel('resultados_predicciones_nn.xlsx')
df_pinn = pd.read_excel('resultados_PINN_esp_final.xlsx')

# Parámetros generales
theta_i = 70
num_indices = 3

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
    if len(valid_indices) >= num_indices:
        break

# RMSE parámetros
param_grupo1 = ['E0', 'C', 'Eg', 'eps_inf']
param_grupo2 = ['A']
etiquetas_grupo1 = [r'$E_0$/eV', r'$C$/eV', r'$E_g$/eV', r'$\varepsilon_\infty$']
etiquetas_grupo2 = [r'$A$']

# Crear figura general (una fila por espectro, 4 columnas)
fig, axs = plt.subplots(nrows=num_indices * 2, ncols=2, figsize=(13, 6.25 * num_indices))
axs = np.array(axs).reshape(num_indices * 2, 2)
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

    # Ψ
    axs[2*row_idx, 0].plot(E_real, np.degrees(psi_real), label='Psi Real', color='blue')
    axs[2*row_idx, 0].plot(E_pred_nn, np.degrees(psi_pred_nn), '--', label='Psi NN', color='orange')
    axs[2*row_idx, 0].plot(E_pred_pinn, np.degrees(psi_pred_pinn), '--', label='Psi PINN', color='green')
    axs[2*row_idx, 0].set_ylabel(r'$\Psi$ (°)', fontsize=label_fontsize)
    axs[2*row_idx, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[2*row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[2*row_idx, 0].legend(fontsize=legend_fontsize)
    axs[2*row_idx, 0].grid(True)

    # Δ
    axs[2*row_idx, 1].plot(E_real, np.degrees(delta_real), label='Delta Real', color='blue')
    axs[2*row_idx, 1].plot(E_pred_nn, np.degrees(delta_pred_nn), '--', label='Delta NN', color='orange')
    axs[2*row_idx, 1].plot(E_pred_pinn, np.degrees(delta_pred_pinn), '--', label='Delta PINN', color='green')
    axs[2*row_idx, 1].set_ylabel(r'$\Delta$ (°)', fontsize=label_fontsize)
    axs[2*row_idx, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[2*row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[2*row_idx, 1].legend(fontsize=legend_fontsize)
    axs[2*row_idx, 1].grid(True)

    # RMSE grupo 1
    rmse_nn1 = [np.sqrt(mean_squared_error([fila_nn[f'{p}_real']], [fila_nn[f'{p}_pred']])) for p in param_grupo1]
    rmse_pinn1 = [np.sqrt(mean_squared_error([fila_pinn[f'{p}_real']], [fila_pinn[f'{p}_pred']])) for p in param_grupo1]
    x1 = np.arange(len(param_grupo1))
    axs[2*row_idx+1, 0].bar(x1 - 0.15, rmse_nn1, 0.3, label='NN', color='lightblue')
    axs[2*row_idx+1, 0].bar(x1 + 0.15, rmse_pinn1, 0.3, label='PINN', color='lightgreen')
    for j, p in enumerate(param_grupo1):
        valor_real = fila_nn[f'{p}_real']
        altura = max(rmse_nn1[j], rmse_pinn1[j]) + 0.01  # para colocar el texto por encima
        axs[2*row_idx+1, 0].text(x1[j], altura, f'{valor_real:.2f}', 
                                ha='center', va='bottom', fontsize=15)

    axs[2*row_idx+1, 0].set_xticks(x1)
    axs[2*row_idx+1, 0].set_xticklabels(etiquetas_grupo1, fontsize=tick_size)
    axs[2*row_idx+1, 0].set_ylabel('RMSE', fontsize=label_fontsize)
    axs[2*row_idx+1, 0].tick_params(axis='both', labelsize=tick_size)
    axs[2*row_idx+1, 0].legend(fontsize=legend_fontsize)
    axs[2*row_idx+1, 0].grid(True)

    # RMSE grupo 2 (solo A)
    rmse_nn2 = [np.sqrt(mean_squared_error([fila_nn[f'A_real']], [fila_nn[f'A_pred']]))]
    rmse_pinn2 = [np.sqrt(mean_squared_error([fila_pinn[f'A_real']], [fila_pinn[f'A_pred']]))]
    axs[2*row_idx+1, 1].bar([-0.15], rmse_nn2, 0.3, label='NN', color='lightblue')
    axs[2*row_idx+1, 1].bar([0.15], rmse_pinn2, 0.3, label='PINN', color='lightgreen')
    for j, p in enumerate(param_grupo2):
        valor_real = fila_nn['A_real']
        altura = max(rmse_nn2[0], rmse_pinn2[0]) + 0.01
        axs[2*row_idx+1, 1].text(0, altura, f'{valor_real:.2f}', 
                                ha='center', va='bottom', fontsize=15)


    axs[2*row_idx+1, 1].set_xticks([0])
    axs[2*row_idx+1, 1].set_xticklabels(etiquetas_grupo2, fontsize=tick_size)
    axs[2*row_idx+1, 1].set_ylabel('RMSE', fontsize=label_fontsize)
    axs[2*row_idx+1, 1].tick_params(axis='both', labelsize=tick_size)
    axs[2*row_idx+1, 1].legend(fontsize=legend_fontsize)
    axs[2*row_idx+1, 1].grid(True)

plt.tight_layout()
plt.savefig('resultados_espectros_tl.png', dpi=300, bbox_inches='tight')
plt.show()


