import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# Define the RNN model
class RNN_MNIST(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10):
        super(RNN_MNIST, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, 28, 28) - treating 28 rows as timesteps, 28 pixels as features
        out, _ = self.lstm(x)  # out: (batch_size, 28, hidden_size)
        out = out[:, -1, :]    # Take the last timestep: (batch_size, hidden_size)
        out = self.relu(self.fc1(out))  # (batch_size, 64)
        out = self.fc2(out)    # (batch_size, num_classes)
        return out

# Training function
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # Inputs: (batch_size, 1, 28, 28) -> (batch_size, 28, 28)
                inputs = inputs.squeeze(1).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# Main function
def main():
    # Data transforms for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Static batch size of 512
    batch_size = 512
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_sizes = {'train': len(trainset), 'val': len(testset)}
    dataloaders = {'train': trainloader, 'val': testloader}

    # Create the model
    model = RNN_MNIST()

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Matching TensorFlow's Adam
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)  # No decay for simplicity

    # Train the model
    num_epochs = 25
    model, history = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs)

    # Plotting
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Loss and Accuracy over Epochs (Static Batch Size 512)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()