import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Define the BasicBlock for ResNet (unchanged)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# Define the ResNet model (unchanged)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Dynamic Batch Size Adjuster
def enhanced_dynamic_batch_size_adjuster(B_t, L_t, G_t, avg_loss, avg_gradient, alpha=20.0, beta=2.0, kappa=100, min_batch=32, max_batch=2048):
    loss_change = (L_t - avg_loss) / (avg_loss + 1e-6)
    grad_change = (G_t - avg_gradient) / (avg_gradient + 1e-6)

    loss_adjustment = alpha * math.tanh(beta * loss_change)
    grad_adjustment = alpha * math.tanh(beta * grad_change)

    batch_size_adjustment = 0
    if abs(loss_adjustment) > 0.1:
        batch_size_adjustment += np.sign(loss_adjustment) * kappa
    if abs(grad_adjustment) > 0.5:
        batch_size_adjustment += np.sign(grad_adjustment) * kappa

    if loss_change < -0.2:
        batch_size_adjustment += 2 * kappa

    B_t_plus_1 = B_t + batch_size_adjustment
    B_t_plus_1 = max(min_batch, min(max_batch, B_t_plus_1))

    return int(B_t_plus_1)

# Training function with DBSA
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs, initial_batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'batch_size': []}

    # DBSA parameters
    batch_size = initial_batch_size
    min_batch, max_batch = 32, 2048
    alpha, beta, kappa = 20.0, 2.0, 100
    avg_loss = 0.0
    avg_gradient = 0.0
    ema_alpha = 0.1

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Adjust dataloader with dynamic batch size
        trainloader = torch.utils.data.DataLoader(
            dataloaders['train'].dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        dataloaders['train'] = trainloader

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for step, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            gradient_magnitude = np.mean([torch.norm(p.grad).item() for p in model.parameters() if p.grad is not None])

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if step == 0 and epoch == 0:
                avg_loss = loss.item()
                avg_gradient = gradient_magnitude
            else:
                avg_loss = (1 - ema_alpha) * avg_loss + ema_alpha * loss.item()
                avg_gradient = (1 - ema_alpha) * avg_gradient + ema_alpha * gradient_magnitude

            batch_size = enhanced_dynamic_batch_size_adjuster(
                batch_size, loss.item(), gradient_magnitude, avg_loss, avg_gradient, alpha, beta, kappa, min_batch, max_batch
            )

            if step % 100 == 0:
                print(f"Step {step}/{len(trainloader)}, Loss: {loss.item():.4f}, Batch Size: {batch_size}")

        scheduler.step()

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / dataset_sizes['val']
        val_epoch_acc = val_corrects.double() / dataset_sizes['val']

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        history['batch_size'].append(batch_size)

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
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

    initial_batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=initial_batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=initial_batch_size, shuffle=False, num_workers=2)

    dataset_sizes = {'train': len(trainset), 'val': len(testset)}
    dataloaders = {'train': trainloader, 'val': testloader}

    # Create the model
    model = ResNet18()

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    # Train the model
    num_epochs = 100
    model, history = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs, initial_batch_size)

    # Plotting
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Loss and Accuracy over Epochs (DBSA)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['batch_size'], label='Batch Size', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Size')
    plt.title('Dynamic Batch Size over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()