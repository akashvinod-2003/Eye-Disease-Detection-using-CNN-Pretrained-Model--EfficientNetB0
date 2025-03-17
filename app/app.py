import os
from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.debug = True

# Ensure directories exist
UPLOAD_FOLDER = os.path.join("static", "uploads")
EXPLANATION_FOLDER = os.path.join("static", "explanations")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPLANATION_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=4)
model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels
classes = ["cataract", "glaucoma", "diabetic_retinopathy", "normal"]

# Function to delete all files in a folder
def cleanup_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete the file
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]

# LIME explanation function
def explain_prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        batch = torch.stack([transform(Image.fromarray(img.astype('uint8'))) for img in images]).to(device)
        return model(batch).detach().cpu().numpy()

    explanation = explainer.explain_instance(
        np.array(image), batch_predict, top_labels=1, num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    
    explanation_img = mark_boundaries(temp / 255.0, mask)
    explain_path = os.path.join(EXPLANATION_FOLDER, os.path.basename(image_path))
    plt.imsave(explain_path, explanation_img)

    return explain_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if 'file' not in request.files:
            return "No file part", 400
            
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
            
        if file and file.filename:
            try:
                # Clean up old files
                cleanup_folder(UPLOAD_FOLDER)
                cleanup_folder(EXPLANATION_FOLDER)

                # Save the new file
                filename = file.filename
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                return "File uploaded successfully", 200
            except Exception as e:
                print(f"Error saving file: {str(e)}")
                return f"Error saving file: {str(e)}", 500
                
    return render_template("index.html")

@app.route("/loading")
def loading():
    # First render the loading page
    if request.args.get('process') != 'true':
        return render_template("loading.html")
    
    try:
        # Get the latest uploaded file
        files = os.listdir(UPLOAD_FOLDER)
        if not files:
            return redirect(url_for('index'))
        
        file_path = os.path.join(UPLOAD_FOLDER, files[0])
        
        # Make prediction and generate explanation
        prediction = predict_image(file_path)
        explanation_path = explain_prediction(file_path)
        
        return redirect(url_for('results', 
                            prediction=prediction, 
                            image_url=file_path, 
                            explanation_url=explanation_path))
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return redirect(url_for('index'))

@app.route("/results")
def results():
    prediction = request.args.get('prediction')
    image_url = request.args.get('image_url')
    explanation_url = request.args.get('explanation_url')
    return render_template("results.html", prediction=prediction, image_url=image_url, explanation_url=explanation_url)

if __name__ == "__main__":
    app.run(debug=True)