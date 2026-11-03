import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SubsetSpeechCommands
import os

# Архітектура легкої нейромережі для IoT-контролера
class AudioCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Вхід: [batch, 1, 64, ~32] (1 канал, 64 mel-фільтри)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        # Після двох шарів MaxPool розмірність зменшується, використовуємо адаптивний розмір для Linear
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
    print("Завантаження даних для навчання...")
    dataset = SubsetSpeechCommands(download=False)
    # Беремо невеликий batch_size для стабільності на CPU
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AudioCNN(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Початок навчання (1 епоха для швидкої генерації моделі)...")
    model.train()
    for i, (specs, labels) in enumerate(dataloader):
          
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print(f"Батч {i+1}/{len(dataloader)}, Втрати (Loss): {loss.item():.4f}")
            
        # Для базового рівня перериваємо після 150 батчів, щоб не чекати довго
       # if i >= 150: 
       #     break

    # Збереження базової моделі
    torch.save(model.state_dict(), "base_model.pth")

    print("\nЗастосування динамічної квантизації...")
    # Оптимізація моделі: перетворення ваг Linear шарів у int8
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    torch.save(quantized_model.state_dict(), "iot_audio_model_quantized.pth")
    print("Квантовану модель збережено!")

    # Порівняння розмірів для фінальної таблиці метрик
    base_size = os.path.getsize("base_model.pth") / 1024
    quant_size = os.path.getsize("iot_audio_model_quantized.pth") / 1024
    
    print("-" * 30)
    print(f"Фізичний розмір базової моделі: {base_size:.2f} КБ")
    print(f"Фізичний розмір оптимізованої моделі: {quant_size:.2f} КБ")
    print("-" * 30)