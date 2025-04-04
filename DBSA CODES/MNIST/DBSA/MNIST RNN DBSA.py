import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import math
import numpy as np

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

# Dynamic Batch Size Adjuster
def enhanced_dynamic_batch_size_adjuster(B_t, L_t, G_t, avg_loss, avg_gradient, alpha=15.0, beta=1.5, kappa=50, min_batch=16, max_batch=2048):
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
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs, initial_batch_size=256):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'batch_size': []}

    # DBSA parameters
    batch_size = initial_batch_size
    min_batch, max_batch = 16, 2048
    alpha, beta, kappa = 15.0, 1.5, 50
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
            inputs = inputs.squeeze(1).to(device)  # (batch_size, 28, 28)
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
            inputs = inputs.squeeze(1).to(device)
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
    # Data transforms for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Initial dataloaders
    initial_batch_size = 256
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=initial_batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=initial_batch_size, shuffle=False, num_workers=2)

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
    model, history = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs, initial_batch_size)

    # Plotting
    epochs_range = range(1, num_epochs + 1)

    # Plot 1: Validation Loss and Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Loss and Accuracy over Epochs (DBSA)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Batch Size Monitoring
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['batch_size'], label='Batch Size', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Size')
    plt.title('Dynamic Batch Size Monitoring over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()