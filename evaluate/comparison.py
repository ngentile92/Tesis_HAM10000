import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Función para cargar resultados desde JSON
def load_evaluation_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Función para graficar F1-Score por clase
def plot_f1_score_comparison(pretrained_results, trained_results, labels):
    f1_pretrained = [pretrained_results["classification_report"][label]["f1-score"] for label in labels]
    f1_trained = [trained_results["classification_report"][label]["f1-score"] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_pretrained, width, label='Preentrenado', color='skyblue')
    bars2 = ax.bar(x + width/2, f1_trained, width, label='Reentrenado', color='orange')

    ax.set_xlabel('Clases')
    ax.set_ylabel('F1-Score')
    ax.set_title('Comparación de F1-Score por Clase')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(os.getcwd(), "f1_score_comparison.png")
    plt.savefig(plot_path)
    print(f"F1-Score Comparison plot saved at {plot_path}")
    plt.close()

# Función para graficar métricas globales
def plot_global_metrics(pretrained_results, trained_results):
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    pretrained_values = [pretrained_results[metric] for metric in metrics]
    trained_values = [trained_results[metric] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, pretrained_values, width, label='Preentrenado', color='skyblue')
    bars2 = ax.bar(x + width/2, trained_values, width, label='Reentrenado', color='orange')

    ax.set_xlabel('Métricas')
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Métricas Globales')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(os.getcwd(), "global_metrics_comparison.png")
    plt.savefig(plot_path)
    print(f"Global Metrics Comparison plot saved at {plot_path}")
    plt.close()

# Función para graficar matriz de confusión
def plot_confusion_matrix(pretrained_results, trained_results, labels):
    pretrained_cm = np.array(pretrained_results["confusion_matrix"])
    trained_cm = np.array(trained_results["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(pretrained_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Matriz de Confusión - Preentrenado")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Verdadero")
    axes[0].set_xticks(np.arange(len(labels)) + 0.5)
    axes[0].set_yticks(np.arange(len(labels)) + 0.5)
    axes[0].set_xticklabels(labels)
    axes[0].set_yticklabels(labels)

    sns.heatmap(trained_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title("Matriz de Confusión - Reentrenado")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Verdadero")
    axes[1].set_xticks(np.arange(len(labels)) + 0.5)
    axes[1].set_yticks(np.arange(len(labels)) + 0.5)
    axes[1].set_xticklabels(labels)
    axes[1].set_yticklabels(labels)

    plt.tight_layout()
    plot_path = os.path.join(os.getcwd(), "confusion_matrix_comparison.png")
    plt.savefig(plot_path)
    print(f"Confusion Matrix Comparison plot saved at {plot_path}")
    plt.close()

# Main
if __name__ == "__main__":
    # Cargar resultados
    pretrained_results = load_evaluation_results("pretrained_model_evaluation_results.json")
    trained_results = load_evaluation_results("trained_model_evaluation_results.json")

    # Etiquetas de clases
    labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    # Generar gráficos
    plot_f1_score_comparison(pretrained_results, trained_results, labels)
    plot_global_metrics(pretrained_results, trained_results)
    plot_confusion_matrix(pretrained_results, trained_results, labels)
