from digitnn import DigitClassifier
from torch import load
from torchvision.transforms import ToTensor
from PIL import Image
from torch import argmax

classifier = DigitClassifier().to('cuda')

# Model prediction
if __name__ == "__main__":
    # Load trained model
    with open('digit_model.pt', 'rb') as file:
        classifier.load_state_dict(load(file))

    # Load and prepare image
    img = Image.open('data\external\img_1.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    # Predict and print the result
    print(argmax(classifier(img_tensor)))
