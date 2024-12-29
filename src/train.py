import numpy as np, cv2, tensorflow as tf, os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
from glob import glob
np.random.seed(42)
tf.random.set_seed(42)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import albumentations as A
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from utils import dice_coef, dice_loss

def conv_block(num_filters, inputs):
    x= Conv2D(num_filters, 3, padding='same')(inputs)
    x= BatchNormalization()(x)
    x= Activation("relu")(x)

    x= Conv2D(num_filters, 3, padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x= conv_block(num_filters, inputs)
    p=MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs, p, num_filters):
    u= Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    print(u.shape,'======', p.shape)
    u= concatenate([u, p])
    c= conv_block(num_filters, u)
    return c

def build_unet(input_shape):
    inputs= Input(input_shape)
    c1, p1= encoder_block(inputs, 16)
    c2, p2= encoder_block(p1, 32)
    c3, p3= encoder_block(p2, 64)
    c4, p4= encoder_block(p3, 128)
    p5= conv_block(256, p4)
    c6=decoder_block(p5, c4, 128)
    c7=decoder_block(c6, c3, 64)
    c8=decoder_block(c7, c2, 32)
    c9=decoder_block(c8, c1, 16)

    outputs= Conv2D(1, 1, padding='same', activation='sigmoid')(c9)
    print(outputs)

    model= Model(inputs, outputs, name='UNET')
    return model



SIZE= (128, 128)
# Define path to save the model in Google Drive

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    images= sorted(glob(os.path.join(path, 'images', '*.png')))
    masks= sorted(glob(os.path.join(path, 'masks', '*.png')))
    print(len(images),'---------------')
    split_size= int(len(images)*split)
    train_x, valid_x= train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y= train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x= train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y= train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)



augmentor = A.Compose([
    A.RandomCrop(width=128, height=128),  # Set to your image size
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2)
])


def augment_image_and_mask(image, mask):
    """Augment both the image and mask using Albumentations."""
    # Convert image and mask to uint8 for Albumentations
    image = (image * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)

    # Apply augmentations
    data = augmentor(image=image, mask=mask)
    augmented_image = data["image"]
    augmented_mask = data["mask"]

    # Convert back to float32 and rescale
    augmented_image = augmented_image.astype(np.float32) / 255.0
    augmented_mask = augmented_mask.astype(np.float32) / 255.0

    return augmented_image, augmented_mask



def tf_parse(x, y):
    def _parse_with_augmentation(image_path, mask_path):
      """Read and augment the image and mask."""
      # Read image and mask
      image_path = image_path.decode()
      mask_path = mask_path.decode()

      image = cv2.imread(image_path, cv2.IMREAD_COLOR)
      image = cv2.resize(image, SIZE)
      image = image / 255.0  # Scale between 0 and 1

      mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
      mask = cv2.resize(mask, SIZE)
      mask = mask / 255.0  # Scale between 0 and 1

      # Apply augmentations
      image, mask = augment_image_and_mask(image, mask)

      # Convert to float32 and expand mask dims
      image = image.astype(np.float32)
      mask = np.expand_dims(mask.astype(np.float32), axis=-1)

      return image, mask

    x, y= tf.numpy_function(_parse_with_augmentation, [x, y], [tf.float32, tf.float32])
    x.set_shape([SIZE[0], SIZE[1], 3])
    y.set_shape([SIZE[0], SIZE[1], 1])
    return x,y

def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__=="__main__":
    #change this to results if you didn't then change the results to results2 in src/test.py
    create_dir("results2")

    #hyperparameter
    batch_size=4
    lr= 1e-4
    num_epochs=500
    model_path=os.path.join("results2", "model.keras")
    csv_path= os.path.join("results2", "log.csv")
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y)=load_dataset('data')

    print("train: {}- {}".format(len(train_x),len(train_y)))
    print("valid: {}- {}".format(len(valid_x),len(valid_y)))
    print("test: {}- {}".format(len(test_x),len(test_y)))


    train_dataset= tf_dataset(train_x, train_y, batch_size)
    valid_dataset= tf_dataset(valid_x, valid_y, batch_size)

    test_dataset= tf_dataset(test_x, test_y, 16)
    ###Model

    model= build_unet((128, 128, 3))
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef])

    callbacks= [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        # EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
    ]

    model.fit(
        train_dataset, epochs=num_epochs,validation_data=valid_dataset,
        callbacks=callbacks
    )