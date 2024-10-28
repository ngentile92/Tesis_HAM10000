# utils/config.py

import yaml
import os
from settings.parameters import PROJECT_ROOT  # Importar PROJECT_ROOT

def load_config(config_path=PROJECT_ROOT / 'settings/config.yaml'):
    if not config_path.exists():
        raise FileNotFoundError(f"El archivo de configuración {config_path} no existe.")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Verificar tipos
    for section, params in config.items():
        for key, value in params.items():
            print(f"Config - {section}.{key}: {value} (type: {type(value)})")
    
    return config


if __name__ == '__main__':
    config = load_config()
    print(config)