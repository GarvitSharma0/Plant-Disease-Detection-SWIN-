from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
from torchvision import transforms

# Load the model from Torch Hub
HUB_URL = "SharanSMenon/swin-transformer-hub"
MODEL_NAME = "swin_tiny_patch4_window7_224"
model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)  # Load from torch hub

# Load the local PyTorch model
model = torch.load('model/model.pt', map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# Define the class labels (replace with your actual class labels)
class_labels = ['Class A', 'Class B', 'Class C', ..., 'Class N']  # Replace with actual labels

# Create the Flask application
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions using the model (POST method)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Open the image file
    image = Image.open(image_file)

    # Preprocess the image
    input_tensor = preprocess(image)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Debugging output size
    print(f"Model output: {output}")

    # Postprocess the output
    try:
        result = postprocess(output)
    except IndexError as e:
        return jsonify({"error": "Predicted index out of range", "details": str(e)}), 500

    return jsonify(result)
def preprocess(image):
    # Define the preprocessing transformations
    preprocess_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Apply transformations to the image and add a batch dimension
    input_tensor = preprocess_transforms(image).unsqueeze(0)
    return input_tensor

class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def postprocess(output):
    # Check the size of the output tensor
    output_size = output.size(1)
    print(f"Output size: {output_size}")

    # Convert the model output to class label
    _, predicted = torch.max(output, 1)
    predicted_index = predicted.item()
    
    # Debugging predicted index
    print(f"Predicted index: {predicted_index}")

    if predicted_index >= len(class_labels) or predicted_index < 0:
        raise IndexError("Predicted index out of range of class labels")

    predicted_class = class_labels[predicted_index]  # Map the index to the class label
    result = {"prediction": predicted_class}
    return result


if __name__ == '__main__':
    app.run(debug=True)
