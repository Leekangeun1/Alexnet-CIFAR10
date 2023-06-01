import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
from AlexNet import AlexNet
from VGG import VGG16

option = input("사용할 모델을 선택하세요.\n 1. AlexNet\n 2. ResNet18\n 3. GoogLeNet\n 4. VGG16\n\n> ")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create an instance of the modified AlexNet model
if (option == '1'):
    print("AlexNet Model Selected:\n")
    # model = models.alexnet(weights="DEFAULT")
    model = AlexNet()
if (option == '2'):
    print("ResNet18 Model Selected:\n")
    model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
if (option == '3'):
    print("GoogLeNet Model Selected:\n")
    model = models.googlenet(weights="GoogLeNet_Weights.IMAGENET1K_V1")
if (option == '4'):
    print("VGG16 Model Selected:\n")
    # model = models.vgg16(weights="DEFAULT")
    print("HELLO")
    model = VGG16()
model.to(device)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate running loss
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    # Print training statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy_train = 100 * correct_train / total_train
    print(
        f"\nEpoch [{epoch + 1}/{epochs}] - Train Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_accuracy_train:.2f}%"
    )

    # Evaluation
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate test accuracy
            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    epoch_accuracy_test = 100 * correct_test / total_test
    print(f"\nEpoch [{epoch + 1}/{epochs}] - Test Accuracy: {epoch_accuracy_test:.2f}%")
