import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from Tauc_Lorentz import generar_nyk, ecuaciones_fresnel, calculo_psi_delta

# Configuración
np.random.seed(42)
n_muestras = 20000
energias_por_muestra = 50
theta_i = 70  # Ángulo de incidencia en grados
E_range = (0.5, 6.5)

# Rango de parámetros (centrados en los valores del ejemplo)
def generar_parametros():
    while True:
        A = np.random.uniform(50, 350)
        E0 = np.random.uniform(1.0, 5.0)
        C = np.random.uniform(0.5, 5.0)
        Eg = np.random.uniform(1.0, 5.0)
        eps_inf = np.random.uniform(1.0, 3.0)


def generar_parametros():
    while True:
        A = np.random.uniform(50, 250)
        E0 = np.random.uniform(1.0, 5.0)
        C = np.random.uniform(1.0, 5.0)
        Eg = np.random.uniform(1.5, 5.5)
        eps_inf = np.random.uniform(1.0, 3.0)

        if E0 <= C / np.sqrt(2):
            continue  # Evita raíz negativa en gamma
        if 4 * E0**2 <= C**2:
            continue  # Evita raíz negativa en alpha
        if E0 == 0 or C == 0:
            continue  # Evita divisiones por cero
        if E0 < 1.0:
            continue
        # Si pasa todas, rompemos el bucle y devolvemos
        break

    return A, E0, C, Eg, eps_inf

# Lista para almacenar los datos
datos = []

for _ in range(n_muestras):
    A, E0, C, Eg, eps_inf = generar_parametros()
    E, n, k, _, _ = generar_nyk(A, E0, C, Eg, eps_inf, Emin=E_range[0], Emax=E_range[1], points=energias_por_muestra)
    rp, rs = ecuaciones_fresnel(n, k, theta_i)
    psi, delta = calculo_psi_delta(rp, rs)

    for i in range(energias_por_muestra):
        datos.append({
            'A': A,
            'E0': E0,
            'C': C,
            'Eg': Eg,
            'eps_inf': eps_inf,
            'E': E[i],
            'psi_deg': np.degrees(psi[i]),
            'delta_deg': np.degrees(delta[i])
        })

# Convertir a DataFrame
df = pd.DataFrame(datos)

# Guardar en CSV
df.to_csv('dataset_elipsometria_10000.csv', index=False)

print("Dataset guardado como 'dataset_elipsometria_10000.csv'")

