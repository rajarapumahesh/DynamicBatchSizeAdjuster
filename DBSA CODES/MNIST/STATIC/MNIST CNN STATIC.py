import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# Define the AllCNN-C model modified for MNIST (1 input channel instead of 3)
class AllCNN_C_MNIST(nn.Module):
    def __init__(self):
        super(AllCNN_C_MNIST, self).__init__()

        self.features = nn.ModuleList([
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1),  # Changed from 3 to 1 input channel
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),  # 10 classes for MNIST
            nn.ReLU()
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

# Function to train the model and collect history
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    # History for plotting
    history = {'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Store validation metrics
            if phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, history

# Main function
def main():
    # Data transforms for MNIST
    transform_train = transforms.Compose([
        transforms.Pad(2),  # MNIST is 28x28, pad to 32x32
        transforms.RandomCrop(28),  # Crop back to 28x28 with random position
        transforms.RandomRotation(10),  # Small rotation for augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    # Create data loaders with batch size 512
    batch_size = 512
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_sizes = {'train': len(trainset), 'val': len(testset)}

    # Create the model
    model = AllCNN_C_MNIST()

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    # Train the model
    num_epochs = 25
    model, history = train_model(model, criterion, optimizer, scheduler,
                               {'train': trainloader, 'val': testloader},
                               dataset_sizes, num_epochs)

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