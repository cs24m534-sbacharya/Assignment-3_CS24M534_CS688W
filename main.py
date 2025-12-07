import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader import get_cifar10
from utils import evaluate
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(epochs=200):
    wandb.init(project="mobilenetv2-cifar10", name="baseline-training")
    trainloader, testloader = get_cifar10(batch_size=128)
    model = models.mobilenet_v2(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        val_acc = evaluate(model, testloader, device)
        wandb.log({"Train Loss": running_loss / len(trainloader), "Train Acc": train_acc, "Val Acc": val_acc, "epoch": epoch})
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "mobilenetv2_cifar10.pth")
    wandb.save("mobilenetv2_cifar10.pth")

if __name__ == '__main__':
    train_model()
