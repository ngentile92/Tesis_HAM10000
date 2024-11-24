import json
import torch
import timm
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from classes.transforms import get_transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import Dataset
import numpy as np
from settings.parameters import CSV_PATH, IMAGES_PATH, TRAINED_MODEL_PATH
# Dataset personalizado
class CustomHAM10000Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, additional_features=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.additional_features = additional_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_id'] + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.dataframe.iloc[idx]['label'], dtype=torch.long)
        if self.additional_features:
            additional_data = [self.dataframe.iloc[idx][feature] for feature in self.additional_features]
            additional_data = torch.tensor(additional_data, dtype=torch.float)
            return (image, *additional_data), label
        return image, label

# Cargar y procesar el DataFrame
df = pd.read_csv(CSV_PATH)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMAGES_PATH, f"{x}.jpg"))
df = df[df['image_path'].apply(os.path.isfile)]
le = LabelEncoder()
df['label'] = le.fit_transform(df['dx'])
scaler = StandardScaler()
df['age'] = scaler.fit_transform(df[['age']].fillna(df['age'].mean()))
le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'].fillna('unknown'))
le_localization = LabelEncoder()
df['localization'] = le_localization.fit_transform(df['localization'].fillna('unknown'))

_, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Transformaciones y DataLoader
transform = get_transforms(augmentation_config={'use': False, 'transformations': []}, input_size=224)
validation_dataset = CustomHAM10000Dataset(val_df, IMAGES_PATH, transform, additional_features=['age', 'sex', 'localization'])
dataloaders = {'val': DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)}

# Función de evaluación con guardado en JSON
def evaluate_model(model, dataloader, device, class_names, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            images, age, sex, localization = inputs
            images = images.to(device)
            age = age.to(device)
            sex = sex.to(device)
            localization = localization.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if hasattr(model, 'additional_fc'):
                additional_features = torch.cat([age.unsqueeze(1), sex.unsqueeze(1), localization.unsqueeze(1)], dim=1)
                additional_out = model.additional_fc(additional_features)
                outputs = torch.cat([outputs, additional_out], dim=1)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()  # Convertir a lista para JSON
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)  # Diccionario para JSON

    # Guardar resultados en un archivo JSON
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }
    with open(f"{model_name}_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics for {model_name} saved to {model_name}_evaluation_results.json")

    return accuracy, precision, recall, f1, conf_matrix, report

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Obtener nombres de clases desde el codificador de etiquetas
    class_names = le.classes_

    # Evaluar modelo preentrenado
    pretrained_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=7)
    pretrained_model = pretrained_model.to(device)
    print("\nEvaluating Pretrained Model:")
    evaluate_model(pretrained_model, dataloaders['val'], device, class_names, model_name="pretrained_model")
    
    # Evaluar modelo entrenado cargado desde el archivo
    trained_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=7)
    trained_model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    trained_model = trained_model.to(device)
    print("\nEvaluating Trained Model:")
    evaluate_model(trained_model, dataloaders['val'], device, class_names, model_name="trained_model")
