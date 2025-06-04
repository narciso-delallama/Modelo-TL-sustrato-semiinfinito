import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def tauc_lorentz_eps2(E, A, E0, C, Eg):
    # Modelo de Tauc-Lorentz para la parte imaginaria de epsilon
    # E: Energía (eV), A: Amplitud, E0: Energía de resonancia, C: Ancho de la línea, Eg: Banda prohibida
    E = np.array(E)
    eps2 = np.zeros_like(E)
    for i, Ei in enumerate(E):
        if Ei > Eg:
            numerator = A * C * E0 * (Ei - Eg)**2
            denominator = ((Ei**2 - E0**2)**2 + C**2 * Ei**2) * Ei
            eps2[i] = numerator / denominator
        else:
            eps2[i] = 0
    # Interpolamos datos teóricos en los mismos puntos que los experimentales
    return eps2

def tauc_lorentz_eps1(E, A, E0, C, Eg, eps_inf=1.0):
    E = np.array(E)
    eps1 = np.full_like(E, eps_inf)
    for i, Ei in enumerate(E):
        try:
            # Definición variables utilizadas
            a_ln = (Eg**2-E0**2)*Ei**2 + (C*Eg)**2 - E0**2*(3*Eg**2 + E0**2)
            a_tan = (Ei**2-E0**2) * (Eg**2+E0**2) + (C*Eg)**2
            alpha = np.sqrt(max(4*E0**2 - C**2, 0))
            gamma = np.sqrt(max(E0**2 - C**2/2, 0))
            psi4 = (Ei**2 - gamma**2)**2 + (alpha * C / 2)**2
            ln1 = np.log((E0**2 + Eg**2 + alpha * Eg) / (E0**2 + Eg**2 - alpha * Eg))
            atan1 = (np.pi - np.arctan((2 * Eg + alpha) / C) + np.arctan((alpha - 2 * Eg) / C))
            # Divisón en términos para evitar simplificar
            t1 = (A * C * a_ln) / (2 * np.pi * psi4 * alpha * E0) * ln1
            t2 = (A * a_tan) / (np.pi * psi4 * E0) * atan1
            t3_inner = (np.arctan((alpha + 2 * Eg) / C)+ np.arctan((alpha - 2 * Eg) / C))
            t3 = (4 * A * E0 * Eg * (Ei**2 - gamma**2) * t3_inner) / (np.pi * psi4 * alpha)
            t4 = (A * E0 * C * (Ei**2 + Eg**2) * np.log(abs((Ei - Eg) / (Ei + Eg)))) / (np.pi * psi4 * Ei)
            t5 = (2 * A * E0 * C * Eg * np.log(abs((Ei - Eg) * (Ei + Eg) / np.sqrt((E0**2 - Eg**2)**2 + Eg**2 * C**2)))) / (np.pi * psi4)
            # Cálculo parte realepsilon real
            eps1[i] += t1 - t2 + t3 - t4 + t5
        except Exception:
            (E0**2 - C**2/2 < 0) or (4*E0**2 - C**2 < 0)
        # Interpolamos datos teóricos en los mismos puntos que los experimentales
    return eps1


def calculo_n_k(eps1, eps2):
    n = np.sqrt((eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)
    k = np.sqrt((-eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)
    return n, k

def generar_nyk(A, E0, C, Eg, eps_inf=1.0, Emin=0.5, Emax=6.5, points=1000):
    E = np.linspace(Emin, Emax, points)
    eps2 = tauc_lorentz_eps2(E, A, E0, C, Eg)
    eps1 = tauc_lorentz_eps1(E, A, E0, C, Eg, eps_inf)
    n, k = calculo_n_k(eps1, eps2)
    return E, n, k, eps1, eps2

def ecuaciones_fresnel(n, k, theta_i):
    Ni = 1.0  # Índice de refracción del aire
    Nt = n + k*1j
    Nti = Nt/Ni
    theta_i = np.radians(theta_i)  # Convertir a radianes
    
    try:
        sqrt_term = np.sqrt(Nti**2 - np.sin(theta_i)**2)
        rp = np.divide(
        (Nti**2 * np.cos(theta_i) - sqrt_term),
        (Nti**2 * np.cos(theta_i) + sqrt_term),
        out=np.zeros_like(sqrt_term),
        where=(Nti**2 * np.cos(theta_i) + sqrt_term) != 0)
        rs = np.divide(
        (np.cos(theta_i) - sqrt_term),
        (np.cos(theta_i) + sqrt_term),
        out=np.zeros_like(sqrt_term),
        where=(np.cos(theta_i) + sqrt_term) != 0)
    except Exception:
        rp = rs = np.nan
    return rp, rs

def calculo_psi_delta(rp, rs):
    rho = np.divide(rp, rs, out=np.zeros_like(rs), where=rs != 0)  # Coeficiente de reflexión
    psi = np.arctan(np.abs(rp) / np.abs(rs))     # Ψ₁
    delta = -np.angle(rho)                       # Δ₁
    return psi, delta

if __name__ == "__main__":
    # ---------- CONFIGURA TAMAÑOS AQUÍ ----------
    title_fontsize = 24
    label_fontsize = 20
    legend_fontsize = 18
    tick_fontsize = 18
    # -------------------------------------------

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from Tauc_Lorentz import generar_nyk, ecuaciones_fresnel, calculo_psi_delta

    # Parámetros del modelo
    Eg, eps_inf, A, E0, C = 2.03, 1.692, 142.599, 3.84, 1.908
    theta_i = 70
    E = np.linspace(0.5, 6.5, 100)

    # Datos generados por el modelo
    E_model, n_model, k_model, eps1_model, eps2_model = generar_nyk(A, E0, C, Eg, eps_inf, Emin=0.5, Emax=6.5, points=100)
    rp_model, rs_model = ecuaciones_fresnel(n_model, k_model, theta_i)
    psi_model, delta_model = calculo_psi_delta(rp_model, rs_model)

    # Datos del archivo psi_delta_inf.txt
    df_psi_delta = pd.read_csv('Psi_Delta_inf.txt', sep='\s+', comment='#')
    E_file = df_psi_delta['eV'].astype(float).values
    psi_file = df_psi_delta['Psi'].astype(float).values
    delta_file = df_psi_delta['Delta'].astype(float).values

    # Datos del archivo eps_inf.txt
    df_eps = pd.read_csv('eps_inf.txt', sep='\s+', comment='#')
    E_eps_file = df_eps['eV'].astype(float).values
    eps1_file = df_eps['Re(Epsilon)'].astype(float).values
    eps2_file = df_eps['Im(Epsilon)'].astype(float).values

    # Crear figura general
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Subfigura 1: Psi y Delta
    axs[0].plot(E_file, psi_file, 'o', label=r'$\Psi$ Soft', lw=2, color='blue')
    axs[0].plot(E_model, np.degrees(psi_model), '-', label=r'$\Psi$ Mod', lw=2, color='red')
    axs[0].plot(E_file, delta_file, 'o', label=r'$\Delta$ Soft', lw=2, color='purple')
    axs[0].plot(E_model, np.degrees(delta_model), '-', label=r'$\Delta$ Mod', lw=2, color='orange')
    axs[0].set_title(r'$\Psi$ y $\Delta$: Software vs Modelo', fontsize=title_fontsize)
    axs[0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[0].set_ylabel('Ángulo (°)', fontsize=label_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].tick_params(axis='both', labelsize=tick_fontsize)
    axs[0].grid(True)

    # Subfigura 2: Epsilon Real e Imaginario
    axs[1].plot(E_eps_file, eps1_file, 'o', label=r'$\varepsilon_r$ Soft', lw=2, color='blue')
    axs[1].plot(E_model, eps1_model, '-', label=r'$\varepsilon_r$ Mod', lw=2, color='red')
    axs[1].plot(E_eps_file, eps2_file, 'o', label=r'$\varepsilon_i$ Soft', lw=2, color='purple')
    axs[1].plot(E_model, eps2_model, '-', label=r'$\varepsilon_i$ Mod', lw=2, color='orange')
    axs[1].set_title(r'$\varepsilon_r$ y $\varepsilon_i$: Software vs Modelo', fontsize=title_fontsize)
    axs[1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[1].set_ylabel(r'Valor de $\varepsilon$', fontsize=label_fontsize)
    axs[1].legend(fontsize=legend_fontsize)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
