import json
import matplotlib.pyplot as plt
import os
from settings.parameters import PROJECT_ROOT  # Importar PROJECT_ROOT

# Cargar datos del entrenamiento
metrics_path = os.path.join(PROJECT_ROOT, "scripts/runs/ham10000_experiment/20epochs-100000-hyper/metrics-5.json")
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

epochs = metrics["epochs"]
train_loss = metrics["train_loss"]
val_loss = metrics["val_loss"]
train_acc = metrics["train_acc"]
val_acc = metrics["val_acc"]

# Gráfico de pérdida
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Pérdida de Entrenamiento", marker='o')
plt.plot(epochs, val_loss, label="Pérdida de Validación", marker='o')
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.title("Evolución de la Pérdida por Épocas")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(os.getcwd(), "loss_progress.png")
plt.savefig(loss_plot_path)
print(f"Gráfico de pérdida guardado como '{loss_plot_path}'.")
plt.show()

# Gráfico de precisión
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label="Precisión de Entrenamiento", marker='o')
plt.plot(epochs, val_acc, label="Precisión de Validación", marker='o')
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.title("Evolución de la Precisión por Épocas")
plt.legend()
plt.grid(True)
plt.tight_layout()
accuracy_plot_path = os.path.join(os.getcwd(), "accuracy_progress.png")
plt.savefig(accuracy_plot_path)
print(f"Gráfico de precisión guardado como '{accuracy_plot_path}'.")
plt.show()

# Cargar datos del reporte de clasificación
classification_report_path = os.path.join(PROJECT_ROOT, "scripts/runs/ham10000_experiment/20epochs-100000-hyper/classification_report_test-5.json")
with open(classification_report_path, 'r') as f:
    classification_report = json.load(f)

labels = list(classification_report.keys())[:-3]  # Excluir métricas globales
f1_scores = [classification_report[label]["f1-score"] for label in labels]

# Gráfico de F1-score por clase
plt.figure(figsize=(10, 6))
plt.bar(labels, f1_scores, color='orange')
plt.xlabel("Clases")
plt.ylabel("F1-Score")
plt.title("F1-Score por Clase en el Conjunto de Prueba")
plt.ylim(0, 1)
for i, score in enumerate(f1_scores):
    plt.text(i, score + 0.02, f'{score:.2f}', ha='center', fontsize=10)
plt.tight_layout()
f1_plot_path = os.path.join(os.getcwd(), "f1_score_test.png")
plt.savefig(f1_plot_path)
print(f"Gráfico de F1-score guardado como '{f1_plot_path}'.")
plt.show()
