FROM python:3.10-slim

# Установка системных зависимостей для сборки некоторых C++ библиотек (нужно для whisperx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Установка зависимостей Python для CPU
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir grpcio grpcio-tools whisperx speechbrain

# Задаем переменную окружения, чтобы кэш моделей сохранялся в определенную папку
ENV HF_HOME=/app/.cache/huggingface

# Копируем исходный код сервера и сгенерированные proto-файлы
COPY server_grpc.py .

# 2. Компилируем proto-файлы прямо внутри контейнера
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. diarization.proto

# Открываем порт для gRPC
EXPOSE 50051

# Команда запуска сервера
CMD ["python", "server_grpc.py"]