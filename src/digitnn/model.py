import torch.nn as nn

# Classifying Digits Neural Network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),  # First convolutional layer
            nn.ReLU(),                 # Activation layer
            nn.Conv2d(32, 64, (3, 3)), # Second convolutional layer
            nn.ReLU(),                 # Activation layer
            nn.Conv2d(64, 64, (3, 3)), # Third convolutional layer
            nn.ReLU(),                 # Activation layer
            nn.Flatten(),              # Flatten the output for the dense layer
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)  # Dense layer with 10 outputs
        )

    def forward(self, x):
        return self.model(x)
