import os
import pandas as pd
from PIL import Image
from settings.parameters import DATA_PATH

class HAM10000DataLoader:
    def __init__(self, base_dir):
        """
        Initialize the data loader with the base directory path.

        :param base_dir: Base directory containing all datasets.
        """
        self.base_dir = base_dir
        self.images_dir = os.path.join(self.base_dir, 'HAM10000_images')
        self.metadata_path = os.path.join(self.base_dir, 'HAM10000_metadata.csv')
        self.segmentations_dir = os.path.join(self.base_dir, 'HAM10000_segmentations_lesion_tschandl')
        self.test_images_dir = os.path.join(self.base_dir, 'ISIC2018_task3_test_images')
        self.ground_truth_path = os.path.join(self.base_dir, 'ISIC2018_Task3_Test_GroundTruth.csv')

    def load_metadata(self):
        """
        Load the metadata file for the HAM10000 dataset.

        :return: DataFrame with metadata.
        """
        return pd.read_csv(self.metadata_path)

    def load_image(self, image_name):
        """
        Load a specific image from the HAM10000 dataset.

        :param image_name: Name of the image file to load.
        :return: PIL Image object.
        """
        image_path = os.path.join(self.images_dir, image_name)
        return Image.open(image_path)

    def load_segmentation(self, segmentation_name):
        """
        Load a segmentation image from the segmentations dataset.

        :param segmentation_name: Name of the segmentation file to load.
        :return: PIL Image object.
        """
        segmentation_path = os.path.join(self.segmentations_dir, segmentation_name)
        return Image.open(segmentation_path)

    def load_test_image(self, image_name):
        """
        Load a test image from the ISIC2018 test dataset.

        :param image_name: Name of the test image file to load.
        :return: PIL Image object.
        """
        test_image_path = os.path.join(self.test_images_dir, image_name)
        return Image.open(test_image_path)

    def load_ground_truth(self):
        """
        Load the ground truth CSV file for the ISIC test set.

        :return: DataFrame with ground truth data.
        """
        return pd.read_csv(self.ground_truth_path)

# Ejemplo de uso:
if __name__ == '__main__':
    data_loader = HAM10000DataLoader(base_dir=DATA_PATH)
    
    # Cargar metadata
    metadata = data_loader.load_metadata()
    print(metadata.head())

    # Cargar una imagen de entrenamiento
    image = data_loader.load_image('ISIC_0024306.jpg')
    image.show()

    # Cargar una imagen de segmentación
    segmentation = data_loader.load_segmentation('ISIC_0024310_segmentation.png')
    segmentation.show()

    # Cargar una imagen de test
    test_image = data_loader.load_test_image('ISIC_0034526.jpg')
    test_image.show()

    # Cargar la ground truth del set de test
    ground_truth = data_loader.load_ground_truth()
    print(ground_truth.head())
