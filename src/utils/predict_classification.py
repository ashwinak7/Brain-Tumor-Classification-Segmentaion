import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import BatchNormalization
from utils.patch_embed import PatchEmbed

# If trained using mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Dummy function to handle 'Cast' layer
def cast_layer(x):
    return tf.cast(x, tf.float32)

def predict_classification(img_path, model_type):
    model_path = None

    if model_type == 'resnet':
        model_path = 'models/Hybrid_ViT.h5'
    elif model_type == 'vit':
        model_path = 'models/Hybrid_ViT.h5'
    else:
        return "Invalid classification model."

    # ✅ Correct custom object as a Lambda layer
    model = tf.keras.models.load_model(model_path,compile=False,custom_objects={"Cast": tf.keras.layers.Lambda(cast_layer)
,"BatchNormalization": tf.keras.layers.BatchNormalization,"PatchEmbed": PatchEmbed})

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    pred_index = np.argmax(preds)
    confidence = float(np.max(preds) * 100)
    classnames = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    return {
        "label": classnames[pred_index],
        "accuracy": f"{confidence:.4f}",}
