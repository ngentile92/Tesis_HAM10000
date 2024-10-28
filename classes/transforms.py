# data/transforms.py

from torchvision import transforms

def get_transforms(augmentation_config, input_size):
    transform_list = [
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Valores estándar de ImageNet
                             std=[0.229, 0.224, 0.225])
    ]
    
    if augmentation_config.get('use', False):
        augmentation_transforms = []
        for aug in augmentation_config.get('transformations', []):
            if aug['name'] == 'RandomHorizontalFlip':
                augmentation_transforms.append(transforms.RandomHorizontalFlip(**aug['params']))
            elif aug['name'] == 'RandomRotation':
                augmentation_transforms.append(transforms.RandomRotation(**aug['params']))
            # Añade más transformaciones según sea necesario
        # Insertar aumentaciones al inicio
        transform_list = augmentation_transforms + transform_list

    return transforms.Compose(transform_list)

