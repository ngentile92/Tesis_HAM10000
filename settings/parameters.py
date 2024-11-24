from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

dict_lesiones = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

DATA_PATH = PROJECT_ROOT / 'data'
CSV_PATH = DATA_PATH / 'HAM10000_metadata.csv'
IMAGES_PATH = DATA_PATH / 'HAM10000_images'
TRAINED_MODEL_PATH = PROJECT_ROOT / 'scripts/runs/ham10000_experiment/20epochs-100000-hyper/best_model-5.pth'