from flask import Flask, jsonify, request
import io
import json
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


app = Flask(__name__)
model = models.googlenet(pretrained=True)
imagenet_class_index = json.load(open('./imagenet_class_index.json'))

  
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)     

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    print(outputs.max(1))
    y_hat = outputs.max(1)[1]
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[y_hat.item()]

    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})
  

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Deploying A Pytorch model with Flask app"
  
if __name__ == '__main__':
    app.run()