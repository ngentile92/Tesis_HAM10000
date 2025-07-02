import os
import json
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from classes.transforms import get_transforms
from settings.parameters import CSV_PATH, IMAGES_PATH, PROJECT_ROOT

# -----------------------------
# Dataset con características opcionales
# -----------------------------
class HAM10000ROCDataset(Dataset):
    """Dataset idéntico al utilizado en entrenamiento pero devuelve imagen y etiqueta."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMAGES_PATH, f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row['label']
        return image, label


def load_data(label_encoder, batch_size=32):
    """Carga el CSV, construye rutas y devuelve DataLoader de TEST (20 % del dataset)."""
    import pandas as pd  # import local para reducir dependencias globales

    # Cargar CSV y filtrar imágenes existentes
    df = pd.read_csv(CSV_PATH)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMAGES_PATH, f"{x}.jpg"))
    df = df[df['image_path'].apply(os.path.isfile)]

    # Codificar etiquetas
    df['label'] = label_encoder.transform(df['dx'])

    # Split estratificado (80/10/10) — usamos solo TEST
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    # Transformaciones (sin augmentaciones)
    transform = get_transforms(augmentation_config={'use': False, 'transformations': []}, input_size=224)

    test_dataset = HAM10000ROCDataset(test_df, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader, len(label_encoder.classes_)


def load_label_encoder(path):
    with open(path, 'r') as f:
        data = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(data['classes'])
    return le


def load_trained_model(num_classes, weights_path, device):
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def compute_multiclass_roc(model, dataloader, n_classes, device):
    """Devuelve FPR, TPR y AUC por clase además de micro y macro."""
    all_labels = []
    all_probs = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true_int = np.concatenate(all_labels, axis=0)

    # Binarizar labels (one-vs-rest)
    y_true = label_binarize(y_true_int, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def plot_roc(fpr, tpr, roc_auc, class_names, save_path="roc_curve_test.png"):
    plt.figure(figsize=(10, 8))

    # Plot macro & micro first
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average ROC (AUC = {roc_auc['micro']:.2f})", color="deeppink", linestyle=":", linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average ROC (AUC = {roc_auc['macro']:.2f})", color="navy", linestyle=":", linewidth=4)

    # Per-class curves
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(class_names)))
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"ROC {class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC multiclase (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"ROC curve saved to {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rutas de artefactos
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "scripts", "models", "final_model.pth")
    LE_PATH = os.path.join(PROJECT_ROOT, "scripts", "utils", "label_encoder.json")
    if not os.path.isfile(WEIGHTS_PATH):
        # fallback al mejor modelo guardado en raíz
        WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
        LE_PATH = os.path.join(PROJECT_ROOT, "scripts", "utils", "label_encoder.json")

    # Cargar codificador de etiquetas
    label_encoder = load_label_encoder(LE_PATH)

    # DataLoader de test
    test_loader, n_classes = load_data(label_encoder)

    # Modelo entrenado
    model = load_trained_model(n_classes, WEIGHTS_PATH, device)

    # Calcular ROC
    fpr, tpr, roc_auc = compute_multiclass_roc(model, test_loader, n_classes, device)

    # Graficar
    class_names = label_encoder.classes_
    plot_roc(fpr, tpr, roc_auc, class_names) 