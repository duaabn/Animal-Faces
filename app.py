import gradio as gr
import requests
from PIL import Image
import numpy as np
import io

# The published API endpoint URL
# Note: We use the direct endpoint for POST requests.
API_URL = "https://duaabn555-animalfacesv2.hf.space/predict_animal"
CONFIDENCE_THRESHOLD = 0.60

def classify_animal(input_image):
    """
    The main function that receives the image from Gradio and communicates 
    with the external API to get a prediction.
    """
    # 1. Prepare Image for HTTP Transmission
    try:
        # Convert to RGB and save to bytes for multipart/form-data upload
        image = input_image.convert("RGB")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Structure the file for the POST request
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        
    except Exception:
        return gr.Markdown("<h2 style='color: #FF6347;'>❌ Invalid image format or processing error.</h2>")

    # 2. Send Prediction Request to External API
    try:
        response = requests.post(API_URL, files=files)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        
    except requests.exceptions.RequestException as e:
        return gr.Markdown(
            f"<h2 style='color: #FF6347; text-align: center;'>❌ API CONNECTION ERROR ❌</h2>"
            f"<p style='text-align: center; color: #555;'>Failed to connect to the external API: {e}</p>"
        )
    
    # 3. Process API Result (Assumes the API returns {"prediction": "...", "confidence": ...})
    predicted_class_name = result.get('prediction', 'Unknown').upper()
    max_confidence = result.get('confidence', 0.0)
    
    # 4. Confidence Threshold Check
    if max_confidence < CONFIDENCE_THRESHOLD:
        return gr.Markdown(
            f"<h2 style='color: #FF6347; text-align: center;'>⚠️ UNCERTAIN CLASSIFICATION ⚠️</h2>"
            f"<p style='text-align: center; color: #555;'>Confidence: {max_confidence:.2%} (Below {CONFIDENCE_THRESHOLD:.0%})</p>"
        )

    # 5. Return Final Result
    return gr.Markdown(
        f"<h1 style='color: #008000; text-align: center; margin-bottom: 5px;'>✅ PREDICTED: {predicted_class_name}</h1>"
        f"<h3 style='text-align: center; margin-top: 5px;'>Confidence: {max_confidence:.2%}</h3>"
    )

# --- Build the Gradio Interface ---
if __name__ == "__main__":
    gr.Interface(
        fn=classify_animal, 
        inputs=gr.Image(type="pil", label="Upload an Animal Image"),
        outputs=gr.Markdown(), 
        title="Animal Classifier (Client Interface)",
        description="This client connects to an external API to classify images. Training details are available in the notebook.",
        live=True,
    ).launch(server_port=7860)
