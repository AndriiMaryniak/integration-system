import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SubsetSpeechCommands

# Точна копія архітектури для завантаження ваг
class AudioCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * 8, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    print("Завантаження датасету для тестування...")
    dataset = SubsetSpeechCommands(download=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    print("Завантаження оптимізованої моделі...")
    model = AudioCNN(num_classes=4)
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(torch.load("iot_audio_model_quantized.pth", weights_only=True))
    model.eval()

    correct = 0
    total = 0

    print("Обчислення точності на всьому наборі даних (це може зайняти хвилину)...")
    with torch.no_grad():
        for specs, labels in dataloader:
            outputs = model(specs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("-" * 30)
    print(f"Точність моделі (Accuracy): {accuracy:.2f}%")
    print("-" * 30)