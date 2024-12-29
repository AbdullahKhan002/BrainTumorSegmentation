import matplotlib.pyplot as plt,os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from utils import dice_coef, dice_loss
from train import load_dataset, tf_dataset

# Load model with custom objects
with tf.keras.utils.custom_object_scope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model("results/model.keras")

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset('data/')

# Load last line from log.csv
log_path = 'results/log.csv'  # Update path as necessary
last_line = pd.read_csv(log_path).iloc[-1]

true_dice = round(last_line['dice_coef'], 2)
predicted_dice = round(last_line['val_dice_coef'], 2)
loss = round(last_line['loss'], 2)
val_loss = round(last_line['val_loss'], 2)

# Prepare y_true and y_pred from segmentation results
y_true = []  # Collect actual masks
y_pred = []  # Collect predicted masks
print('----------------', len(test_y))
for i in range(36):  # Assuming test_x and test_y contain test images and masks
    # Load actual mask
    mask_path = test_y[i]
    actual_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    actual_mask = cv2.resize(actual_mask, (128, 128))  # Resize if needed
    y_true.append(actual_mask.flatten())

    # Load and preprocess image for prediction
    img_path = test_x[i]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict mask
    predicted_mask = model.predict(img, verbose=0)[0]
    predicted_mask = (predicted_mask.squeeze() >= 0.5).astype(np.uint8)  # Binarize
    y_pred.append(predicted_mask.flatten())

# Flatten masks for metrics calculation
y_true = np.concatenate(y_true).astype(int)  # Ensure integer type for metrics
y_pred = np.concatenate(y_pred).astype(int)

# Ensure binary labels
y_true = (y_true > 0).astype(int)  # Convert to binary (0 or 1)
y_pred = (y_pred > 0).astype(int)  # Convert to binary (0 or 1)

# Calculate metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

conf_matrix = confusion_matrix(y_true, y_pred)

# Plot all graphs in one figure
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot Dice Coefficient and Loss
axs[0, 0].bar(['Train Dice', 'Validation Dice'], [true_dice, predicted_dice], color=['blue', 'orange'])
axs[0, 0].set_title('Dice Coefficient')
axs[0, 0].set_ylabel('Value')

axs[0, 1].bar(['Train Loss', 'Validation Loss'], [loss, val_loss], color=['blue', 'orange'])
axs[0, 1].set_title('Loss')
axs[0, 1].set_ylabel('Value')

# Plot Precision, Recall, and F1-Score
axs[1, 0].bar(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1], color='green')
axs[1, 0].set_ylim(0, 1)
axs[1, 0].set_title('Classification Metrics')
axs[1, 0].set_ylabel('Value')

plt.tight_layout()
plt.show()
