# Aplicación de Inteligencia Artificial a la Elipsometría Espectroscópica

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Grado titulado “Aplicación de algoritmos de Inteligencia Artificial a la elipsometría espectroscópica”, cuyo objetivo es predecir parámetros ópticos de películas delgadas a partir de espectros generados o experimentales utilizando redes neuronales.

## Estructura del proyecto

- `dataset_esp.py`  
  Genera un conjunto de 10.000 espectros sintéticos utilizando el modelo Tauc-Lorentz. Incluye graficado interactivo por conjunto de parámetros.

- `nn_esp.py`  
  Entrena una red neuronal tradicional para predecir parámetros ópticos a partir de espectros \((\Psi, \Delta)\). Usa validación cruzada estratificada.

- `PINN_esp.py`  
  Entrena una *Physics-Informed Neural Network* (PINN). Además de minimizar el error en los parámetros, la red debe generar espectros que se ajusten físicamente al modelo Tauc-Lorentz. Esto se logra incorporando una reconstrucción interna de \(\Psi\) y \(\Delta\) en la función de pérdida.

- `graficar2.py`  
  Genera gráficos comparativos de espectros predichos vs reales y errores RMSE por parámetro para muestras seleccionadas.

- `Tauc_Lorentz.py`  
  Implementación completa del modelo Tauc-Lorentz, conversión a índices de refracción \(n\), \(k\), y cálculo de espectros elipsométricos a partir de parámetros.

---


## Paquetes

- Python 3.10.0
- TensorFlow
- Pandas, NumPy, Matplotlib, SciPy, Scikit-learn
- Optuna
- `elli.kkr` para validación con Kramers-Kronig

