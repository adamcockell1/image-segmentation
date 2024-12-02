import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load images and masks
def load_segmentation_data(image_dir, mask_dir, target_size):
    images = []
    masks = []
    for img_name in os.listdir(image_dir):
        # Load and preprocess image
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '_mask.png'))  # Assumes similar naming convention
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)

        # Normalize images and masks
        images.append(image / 255.0)
        masks.append(mask / 255.0)  # Normalize mask to [0, 1]
    
    return np.array(images), np.array(masks).reshape(-1, target_size[0], target_size[1], 1)

# Split dataset
def split_data(images, masks, test_size=0.2):
    return train_test_split(images, masks, test_size=test_size, random_state=42)

# UNet for semantic segmentation
def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)  # Single-channel output for mask

    model = models.Model(inputs, outputs)
    return model

# Train the model
def train_model(model, train_data, val_data, batch_size, epochs):
    train_images, train_masks = train_data
    val_images, val_masks = val_data

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

    history = model.fit(train_images, train_masks,
                        validation_data=(val_images, val_masks),
                        batch_size=batch_size,
                        epochs=epochs)
    return history

# Visualize predictions
def visualize_predictions(images, masks, predictions):
    for i in range(len(images)):
        img = (images[i] * 255).astype(np.uint8)
        true_mask = (masks[i] * 255).astype(np.uint8).squeeze()
        pred_mask = (predictions[i] > 0.5).astype(np.uint8).squeeze() * 255

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(true_mask, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask, cmap='gray')

        plt.show()

# Main workflow
if __name__ == "__main__":
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 16
    EPOCHS = 10

    # Paths
    image_dir = "./dataset/images"
    mask_dir = "./dataset/masks"

    # Load and split data
    images, masks = load_segmentation_data(image_dir, mask_dir, IMG_SIZE)
    train_images, val_images, train_masks, val_masks = split_data(images, masks)

    # Build and train the model
    model = build_unet((IMG_SIZE[0], IMG_SIZE[1], 3))
    history = train_model(model, (train_images, train_masks), (val_images, val_masks), BATCH_SIZE, EPOCHS)

    # Predict and visualize results
    predictions = model.predict(val_images)
    visualize_predictions(val_images, val_masks, predictions)
