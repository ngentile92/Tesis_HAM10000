# training/train.py

import torch
import time
import logging
import json

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs, writer, metrics):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Iniciar el tiempo total de entrenamiento
    training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logging.info(f"Inicio de la época {epoch+1}/{num_epochs}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # Cada época tiene una fase de entrenamiento y validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modo de entrenamiento
                dataloader = dataloaders['train']
            else:
                model.eval()   # Modo de evaluación
                dataloader = dataloaders['val']

            running_loss = 0.0
            running_corrects = 0

            # Iterar sobre los datos
            for inputs, labels in dataloader:
                # Si los inputs son una tupla (imagen y características adicionales)
                if isinstance(inputs, tuple):
                    image, age, sex, localization = inputs
                    image = image.to(device, non_blocking=True)
                    age = age.to(device, non_blocking=True)
                    sex = sex.to(device, non_blocking=True)
                    localization = localization.to(device, non_blocking=True)
                else:
                    # En caso de que solo se utilicen imágenes
                    image = inputs.to(device, non_blocking=True)

                labels = labels.to(device, non_blocking=True)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(image)
                    
                    # Si el modelo tiene la capa adicional para características extra, procesa las características
                    if hasattr(model, 'additional_fc') and isinstance(inputs, tuple):
                        # Combina las características adicionales en un solo tensor y pásalo por `additional_fc`
                        additional_features = torch.cat([age.unsqueeze(1), sex.unsqueeze(1), localization.unsqueeze(1)], dim=1)
                        additional_out = model.additional_fc(additional_features)

                        # Concatenar las salidas de la imagen y las características adicionales antes de la capa final
                        outputs = torch.cat([outputs, additional_out], dim=1)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward y optimización solo en entrenamiento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Estadísticas
                running_loss += loss.item() * image.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # Calcular pérdida y precisión sin usar len()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            logging.info(f"Época {epoch+1}/{num_epochs} - {phase} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Registrar métricas en TensorBoard
            if writer:
                if phase == 'train':
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Acc/train', epoch_acc, epoch)
                else:
                    writer.add_scalar('Loss/val', epoch_loss, epoch)
                    writer.add_scalar('Acc/val', epoch_acc, epoch)

            # Guardar métricas para análisis posterior
            if metrics:
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
                torch.save(best_model_wts, 'models/best_model.pth')
                logging.info(f"Nuevo mejor modelo guardado con Acc: {best_acc:.4f}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logging.info(f"Duración de la época {epoch+1}: {epoch_duration:.2f} segundos")
        print(f"Duración de la época {epoch+1}: {epoch_duration:.2f} segundos")

        # Registrar métricas por época
        if metrics:
            metrics['epochs'].append(epoch + 1)

    logging.info("Entrenamiento completado.")
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time  # Calcular correctamente el tiempo total
    logging.info(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos")
    print(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos")

    # Guardar métricas en un archivo JSON
    if metrics:
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Métricas de entrenamiento guardadas en 'metrics.json'.")

    # Cargar los mejores pesos
    model.load_state_dict(best_model_wts)

    return model
