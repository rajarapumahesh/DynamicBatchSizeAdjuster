import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size=96, hidden_size1=128, hidden_size2=64, num_classes=10):
        super(RNN, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, num_layers=2, dropout=0.3)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True, num_layers=1)
        self.bn = nn.BatchNorm1d(hidden_size2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        # Input x: (batch_size, 32, 32, 3)
        batch_size = x.size(0)
        x = x.view(batch_size, 32, -1)  # Reshape to (batch_size, 32, 96) for sequence processing
        out, _ = self.lstm1(x)  # out: (batch_size, 32, hidden_size1)
        out, _ = self.lstm2(out)  # out: (batch_size, 32, hidden_size2)
        out = out[:, -1, :]  # Take the last timestep: (batch_size, hidden_size2)
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)  # (batch_size, num_classes)
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
                inputs = inputs.to(device)
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
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    batch_size = 256
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_sizes = {'train': len(trainset), 'val': len(testset)}
    dataloaders = {'train': trainloader, 'val': testloader}

    # Create the model
    model = RNN()

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    # Train the model
    num_epochs = 100
    model, history = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs)

    # Plotting
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Loss and Accuracy over Epochs (Static Batch)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()