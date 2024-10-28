# utils/logger.py

import logging
import os

def setup_logger(log_file):
    # Crear el directorio de logs si no existe
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # También añadir logging a la consola
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
