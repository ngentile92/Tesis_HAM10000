# scripts/run_training.py

import os
import time
import json
import logging
from settings.logger import setup_logger
from settings.config import load_config
from settings.parameters import PROJECT_ROOT, CSV_PATH, IMAGES_PATH
from classes.dataset import HAM10000Dataset
from classes.transforms import get_transforms
from models.vision_transformer import create_model
from fit.train import train_model
from fit.evaluate import evaluate_model

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np  # Importar numpy para el muestreo

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directorio creado: {directory}")
        print(f"Directorio creado: {directory}")

def main():
    # Cargar configuración
    config = load_config()

    # Verificar parámetros de muestreo
    print(f"Sample: {config['data'].get('sample', False)}")
    print(f"Sample Size: {config['data'].get('sample_size', 1000)}")

    # Definir las rutas de los directorios
    project_root = os.path.dirname(os.path.abspath(__file__))  # Directorio donde está 'run_training.py'
    models_dir = os.path.join(project_root, 'models')
    utils_dir = os.path.join(project_root, 'utils')
    data_dir = os.path.join(project_root, 'data')
    logs_dir = os.path.join(project_root, 'logs')

    # Asegurar que los directorios existen
    ensure_dir(models_dir)
    ensure_dir(utils_dir)
    ensure_dir(data_dir)
    ensure_dir(logs_dir)

    # Configurar logging
    setup_logger(os.path.join(logs_dir, config['logging']['log_file']))
    logging.info("Inicio del script de entrenamiento y evaluación.")

    # Inicializar TensorBoard
    writer = SummaryWriter(config['tensorboard']['log_dir'])

    # Imprimir el directorio de trabajo actual
    print(f"Directorio de trabajo actual: {os.getcwd()}")

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

    logging.info("Se utiliza todo el conjunto de datos sin muestreo.")

    # Implementar muestreo si está habilitado
    if config['data'].get('sample', False):
        sample_size = config['data'].get('sample_size', 1000)

        # Validar sample_size
        if not isinstance(sample_size, int) or sample_size <= 0:
            error_msg = f"El valor de 'sample_size' debe ser un entero positivo. Valor recibido: {sample_size}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        if len(df) > sample_size:
            # Muestreo aleatorio sin reemplazo
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logging.info(f"Muestreados {sample_size} imágenes del conjunto de datos original.")
            print(f"Muestreados {sample_size} imágenes del conjunto de datos original.")
            print(f"Tamaño del DataFrame después del muestreo: {len(df)}")
        else:
            logging.warning(f"El tamaño del muestreo ({sample_size}) es mayor o igual al tamaño del conjunto de datos actual ({len(df)}). No se realizará muestreo.")
            print(f"El tamaño del muestreo ({sample_size}) es mayor o igual al tamaño del conjunto de datos actual ({len(df)}). No se realizará muestreo.")

    # Definir transformaciones
    transform = get_transforms(
        augmentation_config=config['data']['augmentation'],  # Pasar el diccionario de augmentations
        input_size=config['data']['input_size']
    )
    # Inicializar LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])

    # Número de clases
    num_classes = len(le.classes_)
    logging.info(f"Clases del modelo: {le.classes_}")
    print(f"Clases: {le.classes_}")

    # División del conjunto de datos
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )

    logging.info(f"División del dataset: Entrenamiento={len(train_df)}, Validación={len(val_df)}, Prueba={len(test_df)}")
    print(f"Entrenamiento: {len(train_df)}, Validación: {len(val_df)}, Prueba: {len(test_df)}")

    # Crear instancias de los datasets
    train_dataset = HAM10000Dataset(train_df, transform=transform)
    val_dataset = HAM10000Dataset(val_df, transform=transform)
    test_dataset = HAM10000Dataset(test_df, transform=transform)

    # Crear DataLoaders con optimizaciones
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
    }

    # Definir dataset_sizes como una variable separada
    dataset_sizes = {
        'train': len(train_df),
        'val': len(val_df)
    }

    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Usando dispositivo: {device}")
    print(f"Usando dispositivo: {device}")

    # Crear el modelo
    model = create_model(
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes']
    )
    model.to(device)

    # Definir pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    print(f"Learning Rate: {config['training']['learning_rate']}")
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # Scheduler de decaimiento del aprendizaje
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler_step_size'],
        gamma=config['training']['scheduler_gamma']
    )

    # Variables para registrar tiempos y métricas
    training_start_time = time.time()
    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Entrenar el modelo
    model = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,  # Pasar dataset_sizes como argumento
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['training']['epochs'],
        writer=writer,
        metrics=metrics
    )

    # Evaluar en el conjunto de prueba
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        label_encoder=le,
        phase='test'
    )

    # Guardar el modelo final
    final_model_path = os.path.join(models_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Modelo final guardado en '{final_model_path}'.")
    print(f"Modelo final guardado en '{final_model_path}'.")

    # Guardar el LabelEncoder para uso futuro
    label_encoder_path = os.path.join(utils_dir, 'label_encoder.json')
    with open(label_encoder_path, 'w') as f:
        json.dump({'classes': le.classes_.tolist()}, f, indent=4)
    logging.info(f"LabelEncoder guardado en '{label_encoder_path}'.")
    print(f"LabelEncoder guardado en '{label_encoder_path}'.")

    # Guardar el DataFrame filtrado para referencia
    filtered_dataset_path = os.path.join(data_dir, 'filtered_dataset.csv')
    df.to_csv(filtered_dataset_path, index=False)
    logging.info(f"Dataset filtrado guardado en '{filtered_dataset_path}'.")
    print(f"Dataset filtrado guardado en '{filtered_dataset_path}'.")

    # Guardar las rutas de las imágenes utilizadas
    used_image_paths = df['image_path'].tolist()
    used_image_paths_path = os.path.join(data_dir, 'used_image_paths.txt')
    with open(used_image_paths_path, 'w') as f:
        for path in used_image_paths:
            f.write(f"{path}\n")
    logging.info(f"Rutas de imágenes utilizadas guardadas en '{used_image_paths_path}'.")
    print(f"Rutas de imágenes utilizadas guardadas en '{used_image_paths_path}'.")

    logging.info("Fin del script de entrenamiento y evaluación.")
    print("Fin del script de entrenamiento y evaluación.")

if __name__ == "__main__":
    main()
