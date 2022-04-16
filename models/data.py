from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
train_data = datasets.MNIST(
    root="dataset",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="dataset",
    train=False,
    download=True,
    transform=ToTensor(),
)