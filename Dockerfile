FROM python:3.10-slim

# Установка системных зависимостей (build-essential необходим для сборки C++ расширений whisperx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Шаг 1: Ставим gRPC утилиты отдельно
RUN pip install --no-cache-dir grpcio grpcio-tools

# Шаг 2: Ставим whisperx, speechbrain и форсируем установку только CPU-версий PyTorch.
# Перенос всех тяжелых библиотек в ОДНУ команду не позволит pip скачать 900-мегабайтный CUDA-пакет.
RUN pip install --no-cache-dir \
    whisperx \
    speechbrain \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Задаем переменную окружения для кэша моделей
ENV HF_HOME=/app/.cache/huggingface

# Копируем абсолютно ВСЕ файлы проекта (включая .proto файлы и python-скрипты)
COPY . .

# Компилируем proto-файлы внутри контейнера
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. diarization.proto

EXPOSE 50051

CMD ["python", "server_grpc.py"]