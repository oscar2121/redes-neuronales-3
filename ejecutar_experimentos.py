import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Directorio de trabajo
OUTPUT_DIR = r"C:\Users\oscar\.gemini\antigravity\scratch\redes_neuronales_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fijar semillas para reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def preparar_datos():
    """
    Genera un conjunto de datos desafiante con 30 características y ruido,
    aplica división Train / Dev (Validation) / Test y estandarización sin fuga de datos.
    """
    print("1. Generando dataset de clasificación multidimensional...")
    X, y = make_classification(
        n_samples=3000,
        n_features=30,
        n_informative=18,
        n_redundant=8,
        n_classes=2,
        flip_y=0.07,  # Ruido del 7% en etiquetas para inducir riesgo de overfitting
        random_state=SEED
    )

    # División: 70% Train, 15% Dev (Validation), 15% Test
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    # Preprocesamiento: StandardScaler ajustado estrictamente en TRAIN
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_dev = scaler.transform(X_dev_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"   - Tamaño Conjunto Entrenamiento (Train) : {X_train.shape[0]} muestras (70%)")
    print(f"   - Tamaño Conjunto Validación (Dev)       : {X_dev.shape[0]} muestras (15%)")
    print(f"   - Tamaño Conjunto Prueba (Test)         : {X_test.shape[0]} muestras (15%)")

    # Gráfica 1: Distribución y partición de datos
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Diagrama de pastel / barras de partición
    particiones = ['Entrenamiento\n(Train - 70%)', 'Validación\n(Dev - 15%)', 'Prueba\n(Test - 15%)']
    cantidades = [len(y_train), len(y_dev), len(y_test)]
    colores_pie = ['#333333', '#666666', '#999999']
    axs[0].bar(particiones, cantidades, color=colores_pie, edgecolor='black', width=0.55)
    axs[0].set_title("Distribución de Muestras en Particiones Train/Dev/Test", fontsize=11, fontweight='bold', pad=10)
    axs[0].set_ylabel("Número de Muestras", fontsize=10)
    for i, v in enumerate(cantidades):
        axs[0].text(i, v + 40, f"{v} ({v/len(X)*100:.0f}%)", ha='center', fontweight='bold', fontsize=9)
    axs[0].set_ylim(0, max(cantidades) * 1.15)
    axs[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Distribución de clases por partición (balanceo)
    df_classes = np.array([
        [np.sum(y_train == 0), np.sum(y_train == 1)],
        [np.sum(y_dev == 0), np.sum(y_dev == 1)],
        [np.sum(y_test == 0), np.sum(y_test == 1)]
    ])
    x_pos = np.arange(len(particiones))
    width = 0.3
    axs[1].bar(x_pos - width/2, df_classes[:, 0], width, label='Clase 0', color='#444444', edgecolor='black')
    axs[1].bar(x_pos + width/2, df_classes[:, 1], width, label='Clase 1', color='#888888', edgecolor='black')
    axs[1].set_title("Estratificación y Balance de Clases por Conjunto", fontsize=11, fontweight='bold', pad=10)
    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels(['Train', 'Dev', 'Test'])
    axs[1].set_ylabel("Muestras por Clase", fontsize=10)
    axs[1].legend()
    axs[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    f1_path = os.path.join(OUTPUT_DIR, "fig1_distribucion_datos.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"   [Guardado]: {f1_path}")

    # Convertir a tensores de PyTorch
    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.long)
    t_X_dev = torch.tensor(X_dev, dtype=torch.float32)
    t_y_dev = torch.tensor(y_dev, dtype=torch.long)
    t_X_test = torch.tensor(X_test, dtype=torch.float32)
    t_y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=64, shuffle=True)
    dev_loader = DataLoader(TensorDataset(t_X_dev, t_y_dev), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(t_X_test, t_y_test), batch_size=64, shuffle=False)

    return (t_X_train, t_y_train, t_X_dev, t_y_dev, t_X_test, t_y_test,
            train_loader, dev_loader, test_loader)

# =============================================================================
# ARQUITECTURAS DE REDES NEURONALES
# =============================================================================

class RedBaseNoRegularizada(nn.Module):
    """
    Modelo Base: Red profunda propensa al sobreajuste (High Variance).
    Sin Dropout, sin BatchNorm, sin Weight Decay.
    """
    def __init__(self, input_dim=30, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class RedRegularizadaOptimizada(nn.Module):
    """
    Modelo Optimizado y Regularizado:
    - Batch Normalization en capas ocultas.
    - Dropout (p=0.35) para inhibición de coadaptación.
    - Entrenado con Weight Decay (L2), Optimizador Adam y Scheduler.
    """
    def __init__(self, input_dim=30, num_classes=2, drop_prob=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_prob * 0.7),
            
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# =============================================================================
# CICLOS DE ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def entrenar_modelo(modelo, train_loader, dev_loader, criterion, optimizer,
                    epochs=70, scheduler=None, usar_early_stopping=False, patience=10):
    historial = {
        'train_loss': [], 'dev_loss': [],
        'train_acc': [], 'dev_acc': []
    }
    
    mejor_dev_loss = float('inf')
    mejor_pesos = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Modo Entrenamiento
        modelo.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = modelo(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            preds = torch.argmax(out, dim=1)
            correct_train += (preds == by).sum().item()
            total_train += bx.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Modo Evaluación en DEV
        modelo.eval()
        running_dev_loss = 0.0
        correct_dev = 0
        total_dev = 0

        with torch.no_grad():
            for bx, by in dev_loader:
                out = modelo(bx)
                loss = criterion(out, by)
                running_dev_loss += loss.item() * bx.size(0)
                preds = torch.argmax(out, dim=1)
                correct_dev += (preds == by).sum().item()
                total_dev += bx.size(0)

        epoch_dev_loss = running_dev_loss / total_dev
        epoch_dev_acc = correct_dev / total_dev

        if scheduler is not None:
            scheduler.step()

        historial['train_loss'].append(epoch_train_loss)
        historial['dev_loss'].append(epoch_dev_loss)
        historial['train_acc'].append(epoch_train_acc)
        historial['dev_acc'].append(epoch_dev_acc)

        # Early Stopping
        if usar_early_stopping:
            if epoch_dev_loss < mejor_dev_loss:
                mejor_dev_loss = epoch_dev_loss
                mejor_pesos = copy.deepcopy(modelo.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   [Early Stopping activado en época {epoch}]: Restaurando mejores pesos (dev_loss={mejor_dev_loss:.4f})")
                    modelo.load_state_dict(mejor_pesos)
                    # Rellenar historial para graficar parejo si es necesario
                    break

    return historial

def evaluar_modelo_test(modelo, t_X_test, t_y_test):
    modelo.eval()
    with torch.no_grad():
        logits = modelo(t_X_test)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = t_y_test.cpu().numpy()

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average='binary')
    rec = recall_score(y_true, preds, average='binary')
    f1 = f1_score(y_true, preds, average='binary')
    cm = confusion_matrix(y_true, preds)
    return acc, prec, rec, f1, cm, preds

# =============================================================================
# EJECUCIÓN PRINCIPAL DE EXPERIMENTOS
# =============================================================================

def ejecutar():
    print("="*70)
    print("PRÁCTICA DE REDES NEURONALES III: OPTIMIZACIÓN Y REGULARIZACIÓN")
    print("="*70)

    (t_X_train, t_y_train, t_X_dev, t_y_dev, t_X_test, t_y_test,
     train_loader, dev_loader, test_loader) = preparar_datos()

    criterion = nn.CrossEntropyLoss()
    EPOCHS = 75

    # -------------------------------------------------------------------------
    # EXPERIMENTO 1: MODELO BASE (SIN REGULARIZACIÓN - SGD ESTÁNDAR)
    # -------------------------------------------------------------------------
    print("\n2. Entrenando Modelo Base (Sin regularización, SGD sin momentum)...")
    modelo_base = RedBaseNoRegularizada(input_dim=30)
    # Optimizador clásico sin regularización
    optimizer_base = optim.SGD(modelo_base.parameters(), lr=0.03, momentum=0.0, weight_decay=0.0)
    
    hist_base = entrenar_modelo(
        modelo_base, train_loader, dev_loader, criterion, optimizer_base,
        epochs=EPOCHS, usar_early_stopping=False
    )

    acc_b, prec_b, rec_b, f1_b, cm_b, preds_b = evaluar_modelo_test(modelo_base, t_X_test, t_y_test)
    print(f"   [Base en Test] -> Accuracy: {acc_b:.4f} | Precision: {prec_b:.4f} | Recall: {rec_b:.4f} | F1: {f1_b:.4f}")

    # -------------------------------------------------------------------------
    # EXPERIMENTO 2: MODELO OPTIMIZADO Y REGULARIZADO
    # (Adam + Dropout + Weight Decay L2 + BatchNorm + Early Stopping)
    # -------------------------------------------------------------------------
    print("\n3. Entrenando Modelo Optimizado y Regularizado (Adam + Dropout + L2 + BatchNorm + Early Stopping)...")
    modelo_reg = RedRegularizadaOptimizada(input_dim=30, drop_prob=0.35)
    # Optimizador Adam con L2 Regularization (Weight Decay)
    optimizer_reg = optim.Adam(modelo_reg.parameters(), lr=0.003, weight_decay=1e-4)
    # Learning Rate Scheduler (Reducción por época)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_reg, T_max=EPOCHS, eta_min=1e-5)

    hist_reg = entrenar_modelo(
        modelo_reg, train_loader, dev_loader, criterion, optimizer_reg,
        epochs=EPOCHS, scheduler=scheduler, usar_early_stopping=True, patience=12
    )

    acc_r, prec_r, rec_r, f1_r, cm_r, preds_r = evaluar_modelo_test(modelo_reg, t_X_test, t_y_test)
    print(f"   [Regularizado en Test] -> Accuracy: {acc_r:.4f} | Precision: {prec_r:.4f} | Recall: {rec_r:.4f} | F1: {f1_r:.4f}")

    # -------------------------------------------------------------------------
    # GENERACIÓN DE GRÁFICAS COMPARATIVAS
    # -------------------------------------------------------------------------
    print("\n4. Generando visualizaciones y análisis comparativo...")

    # Gráfica 2: Curvas de Aprendizaje (Pérdida y Precisión)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Pérdida - Modelo Base
    axs[0, 0].plot(hist_base['train_loss'], label='Train Loss', color='#111111', lw=2)
    axs[0, 0].plot(hist_base['dev_loss'], label='Dev (Val) Loss', color='#666666', lw=2, linestyle='--')
    axs[0, 0].set_title("A) Modelo Base: Pérdida (Evidencia de Overfitting / High Variance)", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel("Épocas")
    axs[0, 0].set_ylabel("Cross-Entropy Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Pérdida - Modelo Regularizado
    axs[0, 1].plot(hist_reg['train_loss'], label='Train Loss', color='#111111', lw=2)
    axs[0, 1].plot(hist_reg['dev_loss'], label='Dev (Val) Loss', color='#666666', lw=2, linestyle='--')
    axs[0, 1].set_title("B) Modelo Regularizado: Pérdida Estable (Early Stopping + L2 + Dropout)", fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel("Épocas")
    axs[0, 1].set_ylabel("Cross-Entropy Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Accuracy - Modelo Base
    axs[1, 0].plot(hist_base['train_acc'], label='Train Accuracy', color='#111111', lw=2)
    axs[1, 0].plot(hist_base['dev_acc'], label='Dev (Val) Accuracy', color='#666666', lw=2, linestyle='--')
    axs[1, 0].set_title("C) Modelo Base: Exactitud (Brecha amplia Train vs Dev)", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("Épocas")
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Accuracy - Modelo Regularizado
    axs[1, 1].plot(hist_reg['train_acc'], label='Train Accuracy', color='#111111', lw=2)
    axs[1, 1].plot(hist_reg['dev_acc'], label='Dev (Val) Accuracy', color='#666666', lw=2, linestyle='--')
    axs[1, 1].set_title("D) Modelo Regularizado: Generalización Óptima (Brecha controlada)", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("Épocas")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    f2_path = os.path.join(OUTPUT_DIR, "fig2_curvas_aprendizaje.png")
    plt.savefig(f2_path, dpi=200)
    plt.close()
    print(f"   [Guardado]: {f2_path}")

    # Gráfica 3: Matrices de Confusión Comparativas
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greys', cbar=False, ax=axs[0],
                xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
    axs[0].set_title(f"Matriz de Confusión: Modelo Base\n(Accuracy: {acc_b:.3f} | F1: {f1_b:.3f})", fontsize=11, fontweight='bold')
    axs[0].set_xlabel("Predicción")
    axs[0].set_ylabel("Valor Real")

    sns.heatmap(cm_r, annot=True, fmt='d', cmap='Greys', cbar=False, ax=axs[1],
                xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
    axs[1].set_title(f"Matriz de Confusión: Modelo Regularizado\n(Accuracy: {acc_r:.3f} | F1: {f1_r:.3f})", fontsize=11, fontweight='bold')
    axs[1].set_xlabel("Predicción")
    axs[1].set_ylabel("Valor Real")

    plt.tight_layout()
    f3_path = os.path.join(OUTPUT_DIR, "fig3_matriz_confusion.png")
    plt.savefig(f3_path, dpi=200)
    plt.close()
    print(f"   [Guardado]: {f3_path}")

    # Gráfica 4: Comparativa de Métricas de Evaluación
    fig, ax = plt.subplots(figsize=(9, 5))
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    valores_base = [acc_b, prec_b, rec_b, f1_b]
    valores_reg = [acc_r, prec_r, rec_r, f1_r]

    x = np.arange(len(metricas))
    w = 0.35

    rects1 = ax.bar(x - w/2, valores_base, w, label='Modelo Base (SGD sin Reg)', color='#555555', edgecolor='black')
    rects2 = ax.bar(x + w/2, valores_reg, w, label='Modelo Regularizado (Adam + Dropout + L2)', color='#111111', edgecolor='black')

    ax.set_title("Comparativa de Métricas de Rendimiento en Conjunto de Prueba (Test Set)", fontsize=11, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, fontsize=10)
    ax.set_ylabel("Puntuación (0.0 - 1.0)", fontsize=10)
    ax.set_ylim(0.70, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.005, f"{h:.3f}", ha='center', va='bottom', fontsize=9)
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.005, f"{h:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    f4_path = os.path.join(OUTPUT_DIR, "fig4_comparativa_metricas.png")
    plt.savefig(f4_path, dpi=200)
    plt.close()
    print(f"   [Guardado]: {f4_path}")

    print("\n¡Todos los experimentos completados con éxito!")
    return {
        'acc_base': acc_b, 'prec_base': prec_b, 'rec_base': rec_b, 'f1_base': f1_b,
        'acc_reg': acc_r, 'prec_reg': prec_r, 'rec_reg': rec_r, 'f1_reg': f1_r,
        'cm_base': cm_b.tolist(), 'cm_reg': cm_r.tolist()
    }

if __name__ == "__main__":
    ejecutar()
