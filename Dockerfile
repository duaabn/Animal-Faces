FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files (app.py, notebook, etc.)
COPY . .

EXPOSE 7860

# Command to run the Uvicorn server (which hosts the Gradio app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
