import pandas as pd
import os
from settings.parameters import DATA_PATH, IMAGES_PATH, CSV_PATH
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, confusion_matrix

import time
import json
import logging
import timm
from fit.HAM10000class import HAM10000Dataset
from fit.transformerclass import VisionTransformerClassifier


# Configuración de logging para guardar logs en un archivo
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def main():
    # Inicializar TensorBoard
    writer = SummaryWriter('runs/ham10000_experiment')

    # Iniciar el registro de logs
    logging.info("Inicio del entrenamiento del modelo Vision Transformer en HAM10000 dataset.")

    # Cargar el CSV
    df = pd.read_csv(CSV_PATH)

    # Verificar si todas las imágenes existen
    def image_exists(image_path):
        return os.path.isfile(image_path)
    
    # Crear la columna 'image_path'
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMAGES_PATH, f"{x}.jpg"))

    # Filtrar el DataFrame
    df = df[df['image_path'].apply(image_exists)]

    logging.info(f"Total de muestras válidas después del filtrado: {len(df)}")
    print(f"Total de muestras válidas: {len(df)}")

    # Verificaciones adicionales
    if not os.path.exists(IMAGES_PATH):
        error_msg = f"La carpeta {IMAGES_PATH} no existe."
        logging.error(error_msg)
        raise Exception(error_msg)
    
    specific_image = os.path.join(IMAGES_PATH, 'ISIC_0034320.jpg')
    if not os.path.exists(specific_image):
        error_msg = f"La imagen {specific_image} no existe."
        logging.error(error_msg)
        raise Exception(error_msg)

    # **Eliminar el muestreo del Dataset para usar todo el conjunto de datos**
    # sample_size = 1000  # Número de muestras para la prueba
    # if len(df) > sample_size:
    #     df = train_test_split(df, train_size=sample_size, stratify=df['dx'], random_state=42)[0]
    #     print(f"Dataset reducido a {len(df)} muestras para la prueba inicial.")
    # else:
    #     print(f"El dataset ya tiene {len(df)} muestras, que es igual o menor que el tamaño de muestra especificado.")
    logging.info("Se utiliza todo el conjunto de datos sin muestreo.")

    # Definir transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Valores estándar de ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

    # DATA AUGMENTATION (Descomentar si deseas usar aumentación de datos)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    """

    # Inicializar LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])

    # Número de clases
    num_classes = len(le.classes_)
    logging.info(f"Clases del modelo: {le.classes_}")
    print(f"Clases: {le.classes_}")

    # División del conjunto de datos
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    logging.info(f"División del dataset: Entrenamiento={len(train_df)}, Validación={len(val_df)}, Prueba={len(test_df)}")
    print(f"Entrenamiento: {len(train_df)}, Validación: {len(val_df)}, Prueba: {len(test_df)}")

    # Crear instancias de los datasets
    train_dataset = HAM10000Dataset(train_df, transform=transform)
    val_dataset = HAM10000Dataset(val_df, transform=transform)
    test_dataset = HAM10000Dataset(test_df, transform=transform)

    # Crear DataLoaders con optimizaciones
    batch_size = 32  # Puedes ajustar este valor según la capacidad de tu GPU

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,  # Utilizar la mitad de los núcleos disponibles
        pin_memory=True,
        prefetch_factor=2  # Precarga de datos
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        prefetch_factor=2
    )

    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Usando dispositivo: {device}")
    print(f"Usando dispositivo: {device}")

    # Cargar el modelo preentrenado usando Timm para eficiencia
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model.to(device)

    # Definir pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Scheduler de decaimiento del aprendizaje
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Variables para registrar tiempos y métricas
    training_start_time = time.time()
    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Función de entrenamiento
    def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
        best_model_wts = model.state_dict()
        best_acc = 0.0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logging.info(f"Inicio de la época {epoch+1}/{num_epochs}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('-' * 10)

            # Cada epoch tiene una fase de entrenamiento y validación
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Modo de entrenamiento
                    dataloader = train_loader
                else:
                    model.eval()   # Modo de evaluación
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterar sobre los datos
                for inputs, labels in dataloader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward y optimización solo en entrenamiento
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Estadísticas
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / (len(train_df) if phase == 'train' else len(val_df))
                epoch_acc = running_corrects.double() / (len(train_df) if phase == 'train' else len(val_df))

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                logging.info(f"Época {epoch+1}/{num_epochs} - {phase} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # Registrar métricas en TensorBoard
                if phase == 'train':
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Acc/train', epoch_acc, epoch)
                else:
                    writer.add_scalar('Loss/val', epoch_loss, epoch)
                    writer.add_scalar('Acc/val', epoch_acc, epoch)

                # Guardar métricas para análisis posterior
                if phase == 'train':
                    metrics['train_loss'].append(epoch_loss)
                    metrics['train_acc'].append(epoch_acc.item())
                else:
                    metrics['val_loss'].append(epoch_loss)
                    metrics['val_acc'].append(epoch_acc.item())

                # Guardar el mejor modelo
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    # Guardar el mejor modelo hasta ahora
                    torch.save(best_model_wts, 'best_model.pth')
                    logging.info(f"Nuevo mejor modelo guardado con Acc: {best_acc:.4f}")

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            logging.info(f"Duración de la época {epoch+1}: {epoch_duration:.2f} segundos")
            print(f"Duración de la época {epoch+1}: {epoch_duration:.2f} segundos")

            # Registrar métricas por época
            metrics['epochs'].append(epoch + 1)

        logging.info("Entrenamiento completado.")
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        logging.info(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos")
        print(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos")

        # Guardar métricas en un archivo JSON
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Métricas de entrenamiento guardadas en 'metrics.json'.")

        # Cargar los mejores pesos
        model.load_state_dict(best_model_wts)
        
        # Cerrar el writer de TensorBoard
        writer.close()
        return model

    # Entrenar el modelo
    num_epochs = 10  # Número de epochs para la corrida completa
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

    # Función de evaluación
    def evaluate_model(model, dataloader, phase='test'):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        report = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Guardar el reporte de clasificación en un archivo JSON
        with open(f'classification_report_{phase}.json', 'w') as f:
            json.dump(report, f, indent=4)
        logging.info(f"Reporte de clasificación guardado en 'classification_report_{phase}.json'.")

        # Guardar la matriz de confusión en un archivo CSV
        conf_matrix_df = pd.DataFrame(conf_matrix, index=le.classes_, columns=le.classes_)
        conf_matrix_df.to_csv(f'confusion_matrix_{phase}.csv')
        logging.info(f"Matriz de confusión guardada en 'confusion_matrix_{phase}.csv'.")

        # Imprimir el reporte y la matriz de confusión
        print(f"Reporte de clasificación ({phase}):")
        print(classification_report(all_labels, all_preds, target_names=le.classes_))
        print(f"Matriz de Confusión ({phase}):")
        print(confusion_matrix(all_labels, all_preds))

    # Evaluar en el conjunto de prueba
    evaluate_model(model, test_loader, phase='test')

    # Guardar el modelo final
    torch.save(model.state_dict(), 'final_model.pth')
    logging.info("Modelo final guardado en 'final_model.pth'.")

    logging.info("Fin del script de entrenamiento y evaluación.")

    # Opcional: Guardar el LabelEncoder para uso futuro
    with open('label_encoder.json', 'w') as f:
        json.dump({'classes': le.classes_.tolist()}, f, indent=4)
    logging.info("LabelEncoder guardado en 'label_encoder.json'.")

    # Opcional: Guardar el DataFrame filtrado para referencia
    df.to_csv('filtered_dataset.csv', index=False)
    logging.info("Dataset filtrado guardado en 'filtered_dataset.csv'.")

    # Opcional: Guardar las rutas de las imágenes utilizadas
    used_image_paths = df['image_path'].tolist()
    with open('used_image_paths.txt', 'w') as f:
        for path in used_image_paths:
            f.write(f"{path}\n")
    logging.info("Rutas de imágenes utilizadas guardadas en 'used_image_paths.txt'.")

    # Opcional: Guardar el script de entrenamiento para reproducibilidad
    # Puedes copiar el script actual a un archivo separado si lo deseas.

if __name__ == "__main__":
    main()
