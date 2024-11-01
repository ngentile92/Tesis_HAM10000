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
from itertools import product




def optimize_hyperparameters(config, dataloaders, dataset_sizes, criterion, device, writer):
    # Definir el espacio de hiperparámetros a explorar
    learning_rates = config['hyperparameter_search']['learning_rate_range']
    batch_sizes = config['hyperparameter_search']['batch_size_options']
    scheduler_gammas = config['hyperparameter_search']['scheduler_gamma_range']

    best_acc = 0.0
    best_params = {}
    best_model_state = None

    # Iterar sobre todas las combinaciones posibles
    for lr, batch_size, gamma in product(learning_rates, batch_sizes, scheduler_gammas):
        # Crear modelo
        model = create_model(
            model_name=config['model']['name'],
            pretrained=config['model']['pretrained'],
            num_classes=config['model']['num_classes']
        ).to(device)

        # Crear optimizador y scheduler con los hiperparámetros actuales
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['scheduler_step_size'], gamma=gamma)

        # Actualizar los DataLoaders con el nuevo batch_size
        dataloaders['train'] = DataLoader(dataloaders['train'].dataset, batch_size=batch_size, shuffle=True, num_workers=config['data']['num_workers'])
        dataloaders['val'] = DataLoader(dataloaders['val'].dataset, batch_size=batch_size, shuffle=False, num_workers=config['data']['num_workers'])

        # Entrenar el modelo con estos hiperparámetros
        metrics = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        trained_model = train_model(
            model=model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config['training']['epochs'],
            writer=writer,
            metrics=metrics
        )

        # Evaluar precisión de validación
        val_acc = metrics['val_acc'][-1]  # Última precisión en validación
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = {'learning_rate': lr, 'batch_size': batch_size, 'scheduler_gamma': gamma}
            best_model_state = trained_model.state_dict()

    logging.info(f"Mejores hiperparámetros: {best_params} con una precisión de validación de {best_acc:.4f}")
    torch.save(best_model_state, 'models/best_hyperparameter_model.pth')
    print(f"Modelo con mejores hiperparámetros guardado en 'models/best_hyperparameter_model.pth'")

    return best_params, best_acc


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directorio creado: {directory}")
        print(f"Directorio creado: {directory}")

def main():
    # Load configuration
    config = load_config()

    # Verify sampling parameters
    print(f"Sample: {config['data'].get('sample', False)}")
    print(f"Sample Size: {config['data'].get('sample_size', 1000)}")

    # Define directory paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, 'models')
    utils_dir = os.path.join(project_root, 'utils')
    data_dir = os.path.join(project_root, 'data')
    logs_dir = os.path.join(project_root, 'logs')

    # Ensure directories exist
    ensure_dir(models_dir)
    ensure_dir(utils_dir)
    ensure_dir(data_dir)
    ensure_dir(logs_dir)

    # Setup logging
    setup_logger(os.path.join(logs_dir, config['logging']['log_file']))
    logging.info("Training and evaluation script started.")

    # Initialize TensorBoard
    writer = SummaryWriter(config['tensorboard']['log_dir'])

    # Load CSV data
    df = pd.read_csv(CSV_PATH)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMAGES_PATH, f"{x}.jpg"))
    df = df[df['image_path'].apply(os.path.isfile)]

    logging.info(f"Total valid samples after filtering: {len(df)}")
    print(f"Total valid samples: {len(df)}")

    # Check image path validity
    if not os.path.exists(IMAGES_PATH):
        raise Exception(f"The folder {IMAGES_PATH} does not exist.")
    if not os.path.exists(os.path.join(IMAGES_PATH, 'ISIC_0034320.jpg')):
        raise Exception("Sample image 'ISIC_0034320.jpg' does not exist.")

    # Apply sampling if enabled
    if config['data'].get('sample', False):
        sample_size = config['data'].get('sample_size', 1000)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logging.info(f"Sampled {sample_size} images from the original dataset.")
            print(f"Sampled {sample_size} images from the original dataset.")
        else:
            logging.warning(f"The sample size ({sample_size}) is larger than or equal to the dataset size ({len(df)}). No sampling applied.")

    # Set up transformations
    transform = get_transforms(
        augmentation_config=config['data']['augmentation'],
        input_size=config['data']['input_size']
    )

    # Initialize LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])
    num_classes = len(le.classes_)
    logging.info(f"Model classes: {le.classes_}")
    print(f"Classes: {le.classes_}")

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    logging.info(f"Dataset split: Training={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    print(f"Training: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # Create dataset instances
    train_dataset = HAM10000Dataset(train_df, transform=transform)
    val_dataset = HAM10000Dataset(val_df, transform=transform)
    test_dataset = HAM10000Dataset(test_df, transform=transform)

    # DataLoaders
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    }
    dataset_sizes = {'train': len(train_df), 'val': len(val_df)}

    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    print(f"Using device: {device}")

    # Initialize model, criterion, and optimizer
    model = create_model(model_name=config['model']['name'], pretrained=config['model']['pretrained'], num_classes=num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])

    # Set metrics
    metrics = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if config['hyperparameter_search'].get('use', False):
        print("Hyperparameter optimization enabled.")
        
        # Optimizar los hiperparámetros
        best_params, best_acc = optimize_hyperparameters(config, dataloaders, dataset_sizes, criterion, device, writer)
        logging.info(f"Hyperparameter optimization completed. Best params: {best_params} with validation accuracy {best_acc:.4f}")
        print(f"Best hyperparameters: {best_params} with validation accuracy: {best_acc:.4f}")

        # Crear el modelo usando los mejores parámetros y los valores constantes de `config`
        model = create_model(
            model_name=config['model']['name'],  # Usar `config` como respaldo
            pretrained=config['model']['pretrained'],
            num_classes=num_classes
        )
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['scheduler_step_size'], gamma=best_params['scheduler_gamma'])

    else:
        logging.info("Training with configuration parameters from config.")

    # Train the model
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, config['training']['epochs'], writer, metrics)

    # Evaluate the model on the test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    evaluate_model(model=model, dataloader=test_loader, device=device, label_encoder=le, phase='test')

    # Save the final model and LabelEncoder
    final_model_path = os.path.join(models_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved at '{final_model_path}'.")
    print(f"Final model saved at '{final_model_path}'.")

    label_encoder_path = os.path.join(utils_dir, 'label_encoder.json')
    with open(label_encoder_path, 'w') as f:
        json.dump({'classes': le.classes_.tolist()}, f, indent=4)
    logging.info(f"LabelEncoder saved at '{label_encoder_path}'.")
    print(f"LabelEncoder saved at '{label_encoder_path}'.")

    # Save filtered dataset and used image paths
    filtered_dataset_path = os.path.join(data_dir, 'filtered_dataset.csv')
    df.to_csv(filtered_dataset_path, index=False)
    logging.info(f"Filtered dataset saved at '{filtered_dataset_path}'.")
    print(f"Filtered dataset saved at '{filtered_dataset_path}'.")

    used_image_paths = df['image_path'].tolist()
    used_image_paths_path = os.path.join(data_dir, 'used_image_paths.txt')
    with open(used_image_paths_path, 'w') as f:
        for path in used_image_paths:
            f.write(f"{path}\n")
    logging.info(f"Used image paths saved at '{used_image_paths_path}'.")
    print(f"Used image paths saved at '{used_image_paths_path}'.")

    logging.info("End of training and evaluation script.")
    print("End of training and evaluation script.")

if __name__ == "__main__":
    main()
