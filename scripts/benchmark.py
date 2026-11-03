import requests
import time
import numpy as np

URL = "http://localhost:8000/predict"
# Вкажи шлях до будь-якого тестового файлу (переконайся, що він існує)
TEST_FILE = "data/SpeechCommands/speech_commands_v0.02/down/00176480_nohash_0.wav"

def run_benchmark(num_requests=100, warmup=10):
    print(f"Розігрів сервера ({warmup} запитів)...")
    for _ in range(warmup):
        with open(TEST_FILE, "rb") as f:
            requests.post(URL, files={"file": f})
    
    print(f"Запуск бенчмарку ({num_requests} запитів)...")
    latencies = []
    
    for i in range(num_requests):
        with open(TEST_FILE, "rb") as f:
            start_t = time.perf_counter()
            response = requests.post(URL, files={"file": f})
            end_t = time.perf_counter()
            
            if response.status_code == 200:
                # Беремо End-to-End latency (клієнтську)
                latencies.append((end_t - start_t) * 1000)
            else:
                print(f"Помилка {response.status_code}: {response.text}")
                
    latencies = np.array(latencies)
    print("\n--- Результати Benchmark (Client End-to-End) ---")
    print(f"Mean Latency: {np.mean(latencies):.2f} ms")
    print(f"p50 Latency (Медіана): {np.percentile(latencies, 50):.2f} ms")
    print(f"p95 Latency: {np.percentile(latencies, 95):.2f} ms")

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Помилка з'єднання: {e}. Перевір, чи запущений сервер.")