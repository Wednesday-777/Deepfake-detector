import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "top_activation"


import keras

# Provide backward-compatible Dense layer that ignores quantization_config if not supported
class CompatibleDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


def load_model(model_path: str):
    custom_objects = {"Dense": CompatibleDense}
    try:
        model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    base_model = model.get_layer("efficientnetb0")
    return model, base_model


def _preprocess_image(pil_image: Image.Image):
   
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)  # (224, 224, 3), 0-255
    img_batch = np.expand_dims(img_array, axis=0)
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_batch.copy())
    return img_array, preprocessed


def _make_gradcam_heatmap(img_array_preprocessed, model, base_model,
                           last_conv_layer_name: str = LAST_CONV_LAYER_NAME):
   
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_array_preprocessed)
        x = model.get_layer("global_average_pooling2d")(base_output)
        x = model.get_layer("dropout")(x)
        preds = model.get_layer("dense")(x)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), float(preds.numpy()[0][0])


def _heatmap_to_base64(img_array_original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> str:
    
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()
    heatmap_colored = cm.jet(heatmap_resized)[..., :3]  # RGB, 0-1

    superimposed = heatmap_colored * alpha + (img_array_original / 255.0) * (1 - alpha)
    superimposed = np.clip(superimposed * 255, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(superimposed)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def predict_and_explain(pil_image: Image.Image, model, base_model) -> dict:
    
    img_array_original, img_array_preprocessed = _preprocess_image(pil_image)
    heatmap, pred = _make_gradcam_heatmap(img_array_preprocessed, model, base_model)

    label = "real" if pred > 0.5 else "fake"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    gradcam_b64 = _heatmap_to_base64(img_array_original, heatmap)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "gradcam_image_base64": gradcam_b64,
    }