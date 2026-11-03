import os
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset

# Обмежена підмножина класів згідно з методичкою
TARGET_CLASSES = ['yes', 'no', 'up', 'down']

class SubsetSpeechCommands(Dataset):
    def __init__(self, root="./data", classes=TARGET_CLASSES, download=True):
        os.makedirs(root, exist_ok=True)  # Створює папку, якщо її немає
        # Завантаження базового датасету (відбудеться лише один раз)
        self.dataset = SPEECHCOMMANDS(root, download=download)
        self.classes = classes
        
        # Фільтруємо датасет, залишаючи лише потрібні 4 класи
        self.filtered_data = []
        for filepath in self.dataset._walker:
            label = filepath.split(os.sep)[-2]
            if label in self.classes:
                self.filtered_data.append((filepath, label))
                
        # Налаштування перетворення аудіо на Mel-Spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        filepath, label = self.filtered_data[idx]
        waveform, sample_rate = torchaudio.load(filepath)
        
        # Нормалізація довжини аудіо рівно до 1 секунди (16000 семплів)
        if waveform.shape[1] < 16000:
            pad_amount = 16000 - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > 16000:
            waveform = waveform[:, :16000]

        # Перетворення аудіосигналу у спектрограму (тензор)
        mel_spec = self.mel_transform(waveform)
        
        # Перетворення текстової мітки в індекс (0, 1, 2 або 3)
        label_idx = self.classes.index(label)
        
        return mel_spec, label_idx

if __name__ == "__main__":
    print("Завантаження датасету та ініціалізація...")
    dataset = SubsetSpeechCommands(download=True)
    print(f"Готово! Кількість відфільтрованих аудіофайлів: {len(dataset)}")
    
    spec, label = dataset[0]
    print(f"Розмірність однієї Mel-спектрограми: {spec.shape}")
    print(f"Мітка першого файлу: {label} ({TARGET_CLASSES[label]})")