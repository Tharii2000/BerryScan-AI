from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn

app = FastAPI()

# Define improved custom CNN architecture
class CustomCNN(nn.Module):
    def __init__(self, num_classes=7): #change to 5 to unknown classs
        super(CustomCNN, self).__init__()
        
        # Define the layers of the custom CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(2048)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(2048 * 3 * 3, 4096)  # Adjusted input size
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = self.pool(torch.relu(self.bn6(self.conv6(x))))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

# Load the model
model = CustomCNN(num_classes=7) #change to 5 to unknown classs

# Define the manual transformation
def transform_image(image_bytes):
    # Load the image
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    
    # Convert the image to a tensor
    img_tensor = torch.tensor(np.array(img)).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Change shape from HWC to CHW format and add a batch dimension
    img_tensor = img_tensor.to(device)  # Move tensor to the same device as the model
    return img_tensor

# Load your model here
model.load_state_dict(torch.load('strawbery_disease_model.pth', map_location=torch.device('cpu')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Update class names with treatments
class_names = {
    "0": {
        "name": "angular_leafspot",
        "type": "Bacterial",
        "natural_solution": "Use neem oil spray and avoid overhead watering.",
        "chemical_solution": "Apply copper-based bactericide (e.g., Copper oxychloride)."
    },
    "1": {
        "name": "anthracnose_fruit_rot",
        "type": "Fungal",
        "natural_solution": "Use baking soda and neem oil; prune affected areas.",
        "chemical_solution": "Apply chlorothalonil or mancozeb fungicides."
    },
    "2": {
        "name": "blossom_blight",
        "type": "Fungal",
        "natural_solution": "Improve air flow and use cinnamon or garlic spray.",
        "chemical_solution": "Use captan or thiophanate-methyl fungicides."
    },
    "3": {
        "name": "gray_mold94",
        "type": "Fungal",
        "natural_solution": "Use garlic extract spray; reduce humidity.",
        "chemical_solution": "Apply iprodione or fenhexamid-based fungicides."
    },
    "4": {
        "name": "leaf_spot",
        "type": "Fungal",
        "natural_solution": "Spray with diluted apple cider vinegar or compost tea.",
        "chemical_solution": "Use azoxystrobin or chlorothalonil."
    },
    "5": {
        "name": "powdery_mildew_fruit",
        "type": "Fungal",
        "natural_solution": "Apply milk spray (1:10 ratio) or potassium bicarbonate.",
        "chemical_solution": "Use sulfur fungicides or myclobutanil."
    },
    "6": {
        "name": "powdery_mildew_leaf",
        "type": "Fungal",
        "natural_solution": "Spray with baking soda + vegetable oil + water mix.",
        "chemical_solution": "Apply triadimefon or wettable sulfur."
    }
}

# Endpoint to handle image uploads and predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    tensor = transform_image(image_data)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = str(predicted.item())
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
        
        # Define a confidence threshold
        confidence_threshold = 0.90  # Adjust this value as needed
        
        if confidence < confidence_threshold:
            predicted_class_name = "Unknown"
            Chemical_treatment = "N/A"
            Traditional_treatment = "N/A"
        else:
            predicted_class_name = class_names[predicted_class]["name"]
            Chemical_treatment = class_names[predicted_class]["chemical_solution"]
            Traditional_treatment = class_names[predicted_class]["natural_solution"]
    
    return JSONResponse(content={
        "predicted_class": predicted_class_name,
        "confidence": confidence,
        "chemical_solution": Chemical_treatment,
        "natural_solution": Traditional_treatment
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
