import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from model import *


def load_data(img_dir, mask_dir, target_size):
    '''
    Loads and prepares images for processing by the model

    Parameters:
        img_dir: File system path to image directory
        mask_dir: File system path to image mask directory
        target_size: Target resolution for the image (L, W)

    Returns:
        Array of preprocessed images and masks
    '''
    images = []
    masks = []

    for img_name in Path(img_dir).rglob('*.jpg'):
        img_path = Path(img_dir, img_name)
        mask_path = Path(mask_dir, f'{img_name.stem}_mask.jpg')
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
        image_r = cv2.resize(image, target_size)
        mask_r = cv2.resize(mask, target_size)

        # Normalize images and masks
        images.append(image_r.astype(np.float32) / 255.0)
        masks.append(mask_r.astype(np.float32) / 255.0)

    return np.array(images), np.array(masks).reshape(-1, target_size[0], target_size[1], 1)


def visualize_predictions(images, masks, predictions):
    '''
    Visualizes images and corresponding true vs predicted masks

    Parameters:
        images: Array of original images
        masks: Array of true masks
        predictions: Array of predicted masks
    '''
    for i in range(len(images)):
        img = (images[i] * 255).astype(np.uint8)
        true_mask = (masks[i] * 255).astype(np.uint8).squeeze()
        pred_mask = (predictions[i] > 0.5).astype(np.uint8).squeeze() * 255

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title('True Mask')
        plt.imshow(true_mask, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title('Predicted Mask')
        plt.imshow(pred_mask, cmap='gray')

        plt.show()


if __name__ == '__main__':
    IMG_DIR = Path(Path(__file__).resolve().parents[1], 'data', 'trainingImages')
    MASK_DIR = Path(Path(__file__).resolve().parents[1], 'data', 'trainingMasks')
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 16
    EPOCHS = 20

    # Load and split data
    images, masks = load_data(IMG_DIR, MASK_DIR, IMG_SIZE)
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42)

    # Build and train the model
    unet = build_model()
    history = unet.fit(train_images, train_masks,
                       validation_data=(val_images, val_masks),
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS)

    # Predict and visualize results
    predictions = unet.predict(val_images)
    visualize_predictions(val_images, val_masks, predictions)
