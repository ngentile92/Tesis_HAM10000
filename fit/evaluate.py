# training/evaluate.py

import torch
import json
import logging
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader, device, label_encoder, phase='test'):
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

    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Guardar el reporte de clasificación en un archivo JSON
    with open(f'classification_report_{phase}.json', 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"Reporte de clasificación guardado en 'classification_report_{phase}.json'.")

    # Guardar la matriz de confusión en un archivo CSV
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    conf_matrix_df.to_csv(f'confusion_matrix_{phase}.csv')
    logging.info(f"Matriz de confusión guardada en 'confusion_matrix_{phase}.csv'.")

    # Imprimir el reporte y la matriz de confusión
    print(f"Reporte de clasificación ({phase}):")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    print(f"Matriz de Confusión ({phase}):")
    print(confusion_matrix(all_labels, all_preds))
