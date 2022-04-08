from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.CIFAR100(
    root="raw_data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.CIFAR100(
    root="raw_data",
    train=False,
    download=True,
    transform=ToTensor(),
)