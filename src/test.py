import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from train import load_dataset, tf_dataset
from utils import dice_coef, dice_loss

(train_x, train_y), (valid_x, valid_y), (test_x, test_y)=load_dataset('../data/')
test_dataset= tf_dataset(test_x, test_y, 16)

def segment(image, mask, color):
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    border = dilated_mask - mask
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color[border > 0] = color
    alpha = 0.5
    overlay_image = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    return overlay_image

# Load model with custom objects
with tf.keras.utils.custom_object_scope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model("../results/model.keras")

segmented_actual = []
segmented_y_pred = []

# Iterate over 6 images
for j in range(6):
    # Load and preprocess image
    img_path = test_x[j]
    mask_path = test_y[j]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    x = img / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict mask
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = (y_pred >= 0.5).astype(np.uint8) * 255

    # Segment predicted mask
    segmented_pred = segment(img, y_pred, [0, 255, 0])
    segmented_y_pred.append(segmented_pred)

    # Load and segment actual mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128))
    segmented_truth = segment(img, mask, [255, 0, 0])
    segmented_actual.append(segmented_truth)

# Plot the 6 images in a 3x3 grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(3):
    # Predicted masks
    axes[0, i].imshow(segmented_y_pred[i])
    axes[0, i].set_title(f"Predicted {i+1}")
    axes[0, i].axis("off")

    # Actual masks
    axes[1, i].imshow(segmented_actual[i])
    axes[1, i].set_title(f"Actual {i+1}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
