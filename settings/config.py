# utils/config.py

import yaml
import os
from settings.parameters import PROJECT_ROOT  # Importar PROJECT_ROOT

def load_config(config_path=PROJECT_ROOT / 'settings/config.yaml'):
    # Cargar la configuración desde el archivo YAML
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Verificar tipos y contenido
    for section, params in config.items():
        if not isinstance(params, dict):
            raise TypeError(f"Expected a dictionary for section '{section}', got {type(params)}")
        for key, value in params.items():
            print(f"Config - {section}.{key}: {value} (type: {type(value)})")
    
    return config

if __name__ == '__main__':
    config = load_config()
    print(config)