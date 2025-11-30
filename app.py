import tensorflow as tf
import numpy as np
from PIL import Image
import gradio as gr
import io
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K # Optional: for Keras session cleanup

# --- 1. Configuration ---
MODEL_PATH = './best_efficientnetb0_224_model.keras'
CLASS_NAMES = ["Cat", "Dog", "Wild"]
TARGET_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60 # <--- Minimum confidence score to accept a prediction

# --- 2. Model Loading (Load once) ---
# The model is loaded directly to ensure Gradio can call the prediction function
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    # This error will be raised if loading fails
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

def classify_animal(input_image):
    """
    The main function called by Gradio to process the image, make a prediction,
    and check the confidence threshold.
    
    :param input_image: PIL Image object from Gradio input.
    :return: Gradio Markdown component with the result.
    """
    # 1. Preprocessing
    try:
        # Resize and normalize the image
        image = input_image.resize(TARGET_SIZE)
        img_array = np.array(image, dtype=np.float32) / 255.0
        # Add batch dimension
        processed_image = np.expand_dims(img_array, axis=0)
    except Exception:
        # Return a custom error message if processing fails
        return gr.Markdown("<h2 style='color: #FF6347;'>❌ Invalid image format or processing error.</h2>")

    # 2. Prediction
    # Get the raw prediction probabilities (e.g., [0.1, 0.8, 0.1])
    predictions = MODEL.predict(processed_image)[0]
    max_confidence = float(np.max(predictions))
    predicted_class_index = np.argmax(predictions)
    
    # 3. Confidence Threshold Check
    if max_confidence < CONFIDENCE_THRESHOLD:
        # Return an "uncertain" message (NOT CONFIDENT)
        return gr.Markdown(
            f"<h2 style='color: #FF6347; text-align: center;'>⚠️ UNCERTAIN CLASSIFICATION ⚠️</h2>"
            f"<p style='text-align: center; color: #555;'>The confidence score ({max_confidence:.2%}) is below the required threshold of {CONFIDENCE_THRESHOLD:.0%}.</p>"
            f"<p style='text-align: center; font-weight: bold;'>Result: Unknown Animal or Unclear Image.</p>"
        )

    # 4. Return Final Result
    predicted_class_name = CLASS_NAMES[predicted_class_index].upper()
    
    # Return the successful prediction result formatted with Markdown
    return gr.Markdown(
        f"<h1 style='color: #008000; text-align: center; margin-bottom: 5px;'>✅ PREDICTED: {predicted_class_name}</h1>"
        f"<h3 style='text-align: center; margin-top: 5px;'>Confidence: {max_confidence:.2%}</h3>"
    )

# --- 5. Building the Gradio Interface ---
if __name__ == "__main__":
    gr.Interface(
        fn=classify_animal, 
        inputs=gr.Image(type="pil", label="Upload an Animal Image"),
        outputs=gr.Markdown(), 
        title="EfficientNetB0 Animal Classifier (High Reliability)",
        description=f"Classifies images into {', '.join(CLASS_NAMES)}. Confidence must be above **{CONFIDENCE_THRESHOLD:.0%}** to accept the prediction. The model uses EfficientNetB0.",
        live=True,
    ).launch(server_port=7860)
