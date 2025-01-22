#Import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.optimizers import Adam

# Define dataset directory
base_dir = '<path_to_your_dataset_directory>'


img_size = 256 # Target size for image resizing

# Load training dataset with a small validation split
batch = 32 # Batch size for data loading
train_ds = tf.keras.utils.image_dataset_from_directory(base_dir+'train/',
                                                       seed = 123, # Random seed for reproducibility
                                                       validation_split = 0.01, # 1% of data used for validation
                                                       subset = 'training',
                                                       batch_size = batch,
                                                       image_size = (img_size,img_size))
#Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(base_dir+'train/',
                                                       seed = 123, 
                                                       validation_split = 0.01,
                                                       subset = 'validation',
                                                       batch_size = batch,
                                                       image_size = (img_size,img_size))

# Optimize dataset loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)# Cache, shuffle, and prefetch training data
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)# Prefetch validation data

# Load pre-trained EfficientNetV2S model as the base model
base_model = ConvNeXtLarge(weights = 'imagenet', #Use ImageNet pre-trained weights
                           include_top = False, #Exclude the top classification layer
                           input_shape = (img_size,img_size,3) #Input shape: (target pixels,target pixels,RGB channels)
                           )

# Freeze the base model layers to prevent updates during training, as we are only traning last few layers of the model
base_model.trainable = False

# Define data augmentation pipeline and normalization layer
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ]
)
norm_layer = keras.layers.BatchNormalization()

# Build the final model
model = models.Sequential([
    norm_layer, #normalizing input data
    data_augmentation, #Apply data augmentation
    base_model, #Pretrained EfficientNetV2S
    layers.GlobalAveragePooling2D(), #Pool feature maps
    layers.Dense(512,activation = 'relu'), #Fully connected layer
    layers.Dropout(0.25), #Dropout for regularization
    layers.Dense(40) #Final output layer with 40 classes
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer=Adam(learning_rate = 0.001),#Adam optimizer with a learning rate of 0.001
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Sparse categorical cross-entropy loss
              metrics=['accuracy']) # Track accuracy during training

# Train the model
epochs_size=5 # Number of epochs
history = model.fit(train_ds,validation_data=val_ds,epochs=epochs_size)#stores the accuracy and loss values for each epoch

# Plot training and validation accuracy/loss
epochs_range = range(epochs_size)
plt.figure(figsize=(8,8))
# Plot accuracy
plt.subplot(1,2,2)
plt.plot(epochs_range,history.history['accuracy'],label="Training Accuracy")
plt.plot(epochs_range,history.history['val_accuracy'],label = "Validation Accuracy")
plt.title('Accuracy')
# Plot loss
plt.subplot(1,2,2)
plt.plot(epochs_range,history.history['loss'],label="Training Loss")
plt.plot(epochs_range,history.history['val_loss'],label = "Validation Loss")
plt.title('Loss')

# Define class names for predictions
animal_names=['antelope', 'bat', 'beaver', 'blue+whale', 'bobcat', 'buffalo', 'chihuahua', 'cow', 'dalmatian', 'deer', 'dolphin', 'elephant', 'german+shepherd', 'giant+panda', 'giraffe', 'grizzly+bear', 'hamster', 'hippopotamus', 'humpback+whale', 'killer+whale', 'leopard', 'lion', 'mole', 'mouse', 'otter', 'ox', 'persian+cat', 'pig', 'polar+bear', 'raccoon', 'rat', 'seal', 'siamese+cat', 'skunk', 'spider+monkey', 'tiger', 'walrus', 'weasel', 'wolf', 'zebra']

# Directory for test images
test_ds = base_dir+'test/'
test_predictions = []

# Make predictions on test images
for image in os.listdir(test_ds):
    temp = image # Store image name
    image = test_ds+image # Full path to individual images

    # Load and preprocess the image
    image = tf.keras.utils.load_img(image,target_size=(img_size,img_size))
    img_arr = tf.keras.utils.array_to_img(image) #Convert to array
    img_bat=tf.expand_dims(img_arr,0) #Add batch dimension

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict) #Convert logits to probabilities

    # Append prediction result
    test_predictions.append((temp, animal_names[np.argmax(score)]))

# Save predictions to a CSV file
submission = pd.DataFrame(test_predictions, columns=['image_id', 'class'])
submission.to_csv("predictions.csv", index=False)
