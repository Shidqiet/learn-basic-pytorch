import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import train_data, test_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
num_classes = 10
batch_size = 16
learning_rate = 0.001
epoch = 10

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NeuralNetwork(input_size=input_size, num_classes=num_classes).to(device)

# Dataloader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader =  DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# check accuracy
def check_acc(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
         for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            prediction = model(data)
            _, max_inds = prediction.max(1)
            num_correct += (max_inds == targets).sum()
            num_samples += prediction.size(0)

    print(f"Accuracy on test set: {(num_correct/num_samples)*100:.2f}")

# Train Network
for _ in range(epoch):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # copy data and target to gpu
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        prediction = model(data)
        loss = loss_fn(prediction, targets)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient descent step
        optimizer.step()
    check_acc(test_loader, model)