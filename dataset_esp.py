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
        A = np.random.uniform(50, 250)
        E0 = np.random.uniform(1.0, 5.0)
        C = np.random.uniform(1.0, 5.0)
        Eg = np.random.uniform(1.5, 5.5)
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

# Graficar psi y delta en función de E para los primeros parámetros generados
def graficar_psi_delta_por_indice(df, indice=1):
    """
    Grafica Psi y Delta en función de E para el espectro con el conjunto de parámetros número `indice`.

    Parámetros:
    - df: DataFrame original con columnas ['A', 'C', 'Eg', 'E0', 'eps_inf', 'E', 'psi_deg', 'delta_deg']
    - indice: Índice del conjunto de parámetros únicos (por defecto: 1 → el segundo)
    """
    # Obtener todos los grupos de parámetros únicos
    parametros_unicos = df[['A', 'C', 'Eg', 'E0', 'eps_inf']].drop_duplicates().reset_index(drop=True)

    # Validar índice
    if indice >= len(parametros_unicos):
        raise ValueError(f"Índice fuera de rango. Solo hay {len(parametros_unicos)} conjuntos únicos.")

    # Seleccionar los parámetros deseados
    params = parametros_unicos.iloc[indice]
    subset = df[
        (df['A'] == params['A']) &
        (df['C'] == params['C']) &
        (df['Eg'] == params['Eg']) &
        (df['E0'] == params['E0']) &
        (df['eps_inf'] == params['eps_inf'])
    ].sort_values('E')

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(subset['E'], subset['psi_deg'], label='Ψ (Psi)', color='blue')
    plt.plot(subset['E'], subset['delta_deg'], label='Δ (Delta)', color='orange')
    plt.xlabel('Energía (eV)')
    plt.ylabel('Ángulo (°)')
    plt.title(f'Ψ y Δ vs Energía (conjunto #{indice + 1})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graficar_psi_delta_por_indice(df, indice=1)  # Cambia el índice para graficar otros conjuntos de parámetros
