from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Modified National Institute of Standards and Technology
# Dataset of handwritten digits 0-9
train = datasets.MNIST(root="data", download=True, train=True, transform=transforms.ToTensor())
dataset = DataLoader(train, batch_size=32)  # Loading actual data, chunked in 32 size batches
