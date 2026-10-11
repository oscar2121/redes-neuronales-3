# Redes Neuronales III: Optimización, Regularización y Evaluación de Modelos

Repositorio oficial para la práctica **Redes Neuronales III**, correspondiente a las **Semanas 5, 6 y 7** de la experiencia formativa en **Deep Learning / Percepción Computacional**.

**Estudiante:** Oscar David  
**Usuario de GitHub:** [oscar2121](https://github.com/oscar2121)  
**URL del Repositorio:** [https://github.com/oscar2121/redes-neuronales-3](https://github.com/oscar2121/redes-neuronales-3)

---

## 🎯 Objetivos de la Actividad

1. **Comprensión Conceptual:**
   - Diferenciación formal entre parámetros aprendidos ($W, b$) e hiperparámetros ($\alpha$, *batch size*, épocas, $\lambda$, $p$).
   - Implementación de particiones estratificadas **Train (70%) / Dev (15%) / Test (15%)** bajo rigurosa prevención de **fuga de datos (*Data Leakage*)**.
   - Análisis y diagnóstico del compromiso **Sesgo - Varianza (*Bias-Variance Tradeoff*)**.
   - Evaluación comparativa de algoritmos de optimización: Descenso de Gradiente Estocástico (SGD) vs. **Adam** con *Cosine Annealing LR Scheduler*.
   - Aplicación sinérgica de métodos de regularización: **Dropout**, **Regularización L2 (*Weight Decay*)**, **Batch Normalization** y **Early Stopping**.
   - Medición exhaustiva de desempeño mediante **Accuracy**, **Precision**, **Recall**, **F1-Score** y **Matriz de Confusión**.

2. **Aplicación Práctica:**
   - Desarrollo experimental en **PyTorch** y **TensorFlow / Keras**.
   - Verificación empírica de reducción del error de validación y aumento de la capacidad de generalización en un conjunto de prueba aislado.

---

## 📁 Estructura del Repositorio

```text
├── README.md                                       # Descripción del proyecto, teoría y guía de ejecución
├── 01_optimizacion_y_regularizacion_pytorch.ipynb  # Práctica interactiva en PyTorch (Dropout, L2, Adam, Early Stopping)
├── 02_optimizacion_y_regularizacion_keras.ipynb    # Práctica interactiva en TensorFlow / Keras
├── Redes_Neuronales_III_Practica.ipynb             # Cuaderno maestro con benchmark comparativo para Google Colab
├── ejecutar_experimentos.py                        # Script modular y reproducible de entrenamiento
├── Redes_Neuronales_III_Informe.pdf                # Informe oficial formal en PDF (entrega institucional)
├── Redes_Neuronales_III_Informe.docx               # Informe editable en Microsoft Word
└── figuras/                                        # Evidencias gráficas de entrenamiento y métricas
    ├── fig1_distribucion_datos.png                 # Distribución estratificada Train/Dev/Test
    ├── fig2_curvas_aprendizaje.png                 # Curvas de pérdida y exactitud (Sobreajuste vs. Regularización)
    ├── fig3_matriz_confusion.png                   # Matrices de confusión en Test Set
    └── fig4_comparativa_metricas.png               # Comparativa de métricas en Test Set
```

---

## 📊 Resultados Experimentales en Conjunto de Prueba (*Test Set*)

Evaluación realizada sobre 450 muestras completamente aisladas:

| Métrica de Evaluación | Modelo Base (SGD plano, Sin Regularización) | Modelo Regularizado (Adam + L2 + Dropout + Early Stopping) | Mejora Absoluta |
| :--- | :---: | :---: | :---: |
| **Accuracy (Exactitud)** | 92.22% | **94.67%** | **+2.45%** |
| **Precision (Precisión)** | 92.11% | **94.71%** | **+2.60%** |
| **Recall (Sensibilidad)** | 92.51% | **94.71%** | **+2.20%** |
| **F1-Score (Media Armónica)** | 92.31% | **94.71%** | **+2.40%** |
| **Pérdida Mínima en Validación** | 0.3542 *(Divergente / Alta Varianza)* | **0.2005 *(Convergente en época 36)*** | **-43.4% en error** |

---

## 🚀 Ejecución en Google Colab

Puedes abrir y ejecutar los cuadernos directamente en Google Colab haciendo clic en los siguientes enlaces:

- 📓 [Abrir Cuaderno Maestro en Colab](https://colab.research.google.com/github/oscar2121/redes-neuronales-3/blob/main/Redes_Neuronales_III_Practica.ipynb)
- 📓 [Abrir Práctica PyTorch en Colab](https://colab.research.google.com/github/oscar2121/redes-neuronales-3/blob/main/01_optimizacion_y_regularizacion_pytorch.ipynb)
- 📓 [Abrir Práctica Keras en Colab](https://colab.research.google.com/github/oscar2121/redes-neuronales-3/blob/main/02_optimizacion_y_regularizacion_keras.ipynb)

---

## 💻 Ejecución Local

Para clonar y ejecutar localmente:

```bash
git clone https://github.com/oscar2121/redes-neuronales-3.git
cd redes-neuronales-3
pip install torch torchvision scikit-learn matplotlib seaborn reportlab python-docx
python ejecutar_experimentos.py
```
