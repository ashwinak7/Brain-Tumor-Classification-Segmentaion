import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report
from tensorflow import keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import AdamW
from sklearn.metrics import f1_score
tf.keras.backend.clear_session()
mixed_precision.set_global_policy("mixed_float16")
# **Dataset Paths**
# Set dataset paths
train_dir = "/kaggle/input/bt-dataset/Classes/Training"
test_dir = "/kaggle/input/bt-dataset/Classes/Testing"
img_size = (224,224)
batch_size = 32  # Reduce to 4 if needed
datagen = ImageDataGenerator(
    featurewise_center=True,       # Normalize per channel
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)
datagen = ImageDataGenerator(
    featurewise_center=True,       # Normalize per channel
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)
def load_dataset(directory):
    images, labels = [], []
    class_names = sorted(os.listdir(directory))
    class_map = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0
            images.append(img)
            labels.append(class_map[class_name])
    
    return np.array(images), np.array(labels), class_map
# Load train and test data
x_train, y_train, class_map = load_dataset(train_dir)
x_test, y_test, _ = load_dataset(test_dir)

datagen.fit(x_train)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, len(class_map))
y_test = keras.utils.to_categorical(y_test, len(class_map))    
datagen = ImageDataGenerator(
    featurewise_center=True,       # Normalize per channel
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)
# Patch Embedding Layer
class PatchEmbed(layers.Layer):
    def __init__(self, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.projection = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)

    def call(self, x):
        x = self.projection(x)  # Convert image into patches
        x = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))  # Flatten patches
        return x

# Vision Transformer Block with L2 Regularization
def vit_block(x, num_heads=8, embed_dim=256):
    x = layers.LayerNormalization()(x)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, kernel_regularizer=l2(0.0005))(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.Dropout(0.3)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(embed_dim, activation="relu", kernel_regularizer=l2(0.0005))(x)
    return x
# **Hybrid ViT**
# CNN Backbone

# CNN Backbone with L2 Regularization
def cnn_backbone(inputs):
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005))(inputs)
    x = layers.BatchNormalization()(x)  # Add BatchNorm
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005))(x)
    x = layers.BatchNormalization()(x)  # Add BatchNorm
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005))(x)
    x = layers.BatchNormalization()(x)  # Add BatchNorm
    x = layers.MaxPooling2D((2, 2))(x)
    
    return x

# Hybrid ViT-CNN Model
def build_hybrid_vit(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # CNN Feature Extraction
    # CNN Backbone
    # CNN Feature Extraction
    cnn_features = cnn_backbone(inputs)  # Keep (batch, H, W, C) format

    # Patch Embedding
    patches = PatchEmbed()(cnn_features)
    
    # Transformer Blocks (ViT)
    vit_out = vit_block(patches)
    vit_out = layers.GlobalAveragePooling1D()(vit_out)
    
    # Final Classification Head
    x = layers.Dense(64, activation="relu")(vit_out)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)



# Build & Compile Model
model = build_hybrid_vit((224, 224, 3), num_classes=len(class_map))
optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# **Compiling**
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Compile and train the model & Learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=batch_size,shuffle=True,callbacks=[lr_scheduler, early_stopping])

# Print available history keys
print(history.history.keys())
# Evaluate the model
model.evaluate(x_test, y_test)

# Plot the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predict on test data
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true_labels, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predict on test data
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Generate classification report
report = classification_report(y_true_labels, y_pred_labels, target_names=class_map.keys(), digits=4)
print(report)

# **Dice Score Calculation**
# Function to calculate Dice coefficient
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)  # Avoid division by zero

# Convert predictions to binary format (one-hot encoded)
y_pred_bin = (y_pred > 0.5).astype(int)
y_true_bin = y_test.astype(int)

# Calculate Dice coefficient for each class
dice_scores = [dice_coefficient(y_true_bin[:, i], y_pred_bin[:, i]) for i in range(len(class_map))]

# Print Dice scores for each class
for class_name, score in zip(class_map.keys(), dice_scores):
    print(f"Dice Score for {class_name}: {score:.4f}")

# Compute overall F1-score (similar to Dice coefficient for multi-class)
f1_macro = f1_score(y_true_labels, y_pred_labels, average="macro")
print(f"Overall Dice Score (F1-macro): {f1_macro:.4f}")

# **Visualizing**
# Reverse class_map to get label names
class_names = {v: k for k, v in class_map.items()}

# Number of images per class to display
num_images_per_class = 5

# Find indices for each class
glioma_indices = [i for i in range(len(y_true_labels)) if y_true_labels[i] == class_map["glioma"]][:num_images_per_class]
meningioma_indices = [i for i in range(len(y_true_labels)) if y_true_labels[i] == class_map["meningioma"]][:num_images_per_class]
pituitary_indices = [i for i in range(len(y_true_labels)) if y_true_labels[i] == class_map["pituitary"]][:num_images_per_class]
no_tumor_indices = [i for i in range(len(y_true_labels)) if y_true_labels[i] == class_map["notumor"]][:num_images_per_class]

# Combine all indices
selected_indices = glioma_indices + meningioma_indices + pituitary_indices + no_tumor_indices

# Set up figure
plt.figure(figsize=(15, 10))

# Visualizing selected images
for i, idx in enumerate(selected_indices):
    plt.subplot(4, num_images_per_class, i + 1)
    plt.imshow(x_test[idx])  # Display the image
    true_label = class_names[y_true_labels[idx]]
    predicted_label = class_names[y_pred_labels[idx]]
    plt.title(f"Truth: {true_label}\nPred: {predicted_label}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
