from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from fastapi.responses import HTMLResponse, JSONResponse
import torch
from torchvision import transforms
import os
import pickle

# Create a FastAPI application
app = FastAPI()

# Define an endpoint to serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def get_prediction_page():
    with open("predict.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Load the pre-trained model
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)

        self.fc1 = torch.nn.Linear(in_features=48 * 12 * 12, out_features=240)
        self.fc2 = torch.nn.Linear(in_features=240, out_features=120)
        self.out = torch.nn.Linear(in_features=120, out_features=38)

    def forward(self, t):
        t = t

        t = self.conv1(t)
        t = torch.nn.functional.relu(t)
        t = torch.nn.functional.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = torch.nn.functional.relu(t)
        t = torch.nn.functional.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3(t)
        t = torch.nn.functional.relu(t)
        t = torch.nn.functional.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv4(t)
        t = torch.nn.functional.relu(t)
        t = torch.nn.functional.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 48 * 12 * 12)
        t = self.fc1(t)
        t = torch.nn.functional.relu(t)

        t = self.fc2(t)
        t = torch.nn.functional.relu(t)

        t = self.out(t)

        return t

model = Network()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()  

# Load the label mapping
with open('labels.pkl', 'rb') as f:
    reference = pickle.load(f)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define an endpoint to accept image uploads
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the contents of the uploaded file
        contents = await file.read()
        # Convert the image bytes to a PIL Image object
        image = Image.open(io.BytesIO(contents))
        # Preprocess the image
        image = transform(image).unsqueeze(0)
        # Make predictions using the loaded model
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            # Map the predicted index to the corresponding label
            predicted_label = list(reference.keys())[list(reference.values()).index(predicted.item())]
            # Return the predicted class label as JSON response
            return JSONResponse(content={"predicted_class": predicted_label})
    except Exception as e:
        # If an exception occurs, return an HTTP 500 Internal Server Error with the error message
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#uvicorn main:app --reload
