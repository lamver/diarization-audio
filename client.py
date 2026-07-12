import grpc
import os
import time

# Импортируем сгенерированные файлы контракта
import diarization_pb2
import diarization_pb2_grpc

def generate_chunks(audio_path, chunk_size=64 * 1024):
    """Генератор, который читает файл кусками по 64 КБ для отправки в gRPC Stream"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Аудиофайл не найден по пути: {audio_path}")
        
    print(f"Открытие файла {audio_path} для отправки...")
    with open(audio_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            # Оборачиваем байты в структуру, описанную в .proto файле
            yield diarization_pb2.AudioChunk(data=chunk)

def run():
    # Путь к тестовому аудиофайлу
    audio_file = "meeting.wav" 
    server_address = "ms-24.aisearch.tech:50051"

    # Настраиваем gRPC канал.
    # Так как обработка на CPU занимает время, увеличиваем лимиты таймаутов, 
    # чтобы соединение не порвалось во время работы нейросетей.
    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024), # 100 МБ макс ответ
        ('grpc.max_send_message_length', 100 * 1024 * 1024)     # 100 МБ макс запрос
    ]
    
    print(f"Подключение к gRPC серверу {server_address}...")
    with grpc.insecure_channel(server_address, options=options) as channel:
        
        # АВТООПРЕДЕЛЕНИЕ КЛИЕНТСКОГО КЛАССА (STUB):
        # Защищает от разницы версий компилятора grpcio-tools на Windows
        if hasattr(diarization_pb2_grpc, "DiarizationServiceStub"):
            stub = diarization_pb2_grpc.DiarizationServiceStub(channel)
        elif hasattr(diarization_pb2_grpc, "DiarizationService"):
            stub = diarization_pb2_grpc.DiarizationService(channel)
        else:
            raise AttributeError("Не удалось найти клиентский gRPC класс в diarization_pb2_grpc.py")
        
        start_time = time.time()
        try:
            # Создаем итератор чанков аудиофайла
            audio_stream = generate_chunks(audio_file)
            
            print("Отправка аудио в стрим. Ожидайте, нейросети на CPU обрабатывают файл...")
            # Вызываем метод gRPC. Он заблокирует выполнение, пока сервер не пришлет финальный ответ
            response = stub.ProcessAudio(audio_stream)
            
            print(f"\n--- Обработка успешно завершена за {time.time() - start_time:.2f} сек. ---")
            print(f"Статус сервера: {response.status}\n")
            print("Полученный протокол встречи:")
            
            # Выводим информацию о каждом сегменте
            for index, segment in enumerate(response.segments, 1):
                # Ограничиваем вывод вектора первыми 3 числами, чтобы не спамить консоль (всего их 192)
                vector_preview = [round(num, 4) for num in segment.embedding[:3]]
                
                print(f"Реплика №{index}")
                print(f"  ВРЕМЯ:   [{segment.start:05.2f} - {segment.end:05.2f}] сек.")
                print(f"  СПИКЕР:  {segment.speaker}")
                print(f"  ТЕКСТ:   {segment.text}")
                print(f"  ВЕКТОР:  {vector_preview}... (Всего чисел: {len(segment.embedding)})")
                print("-" * 50)
                
        except grpc.RpcError as e:
            print(f"\nОшибка gRPC соединения: {e.code()}")
            print(f"Детали: {e.details()}")
        except Exception as e:
            print(f"\nНепредвиденная ошибка клиента: {e}")

if __name__ == "__main__":
    run()