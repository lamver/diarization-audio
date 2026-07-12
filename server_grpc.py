import grpc
from concurrent import futures
import io
import os
import gc
import whisperx
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

import diarization_pb2
import diarization_pb2_grpc

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
# Считываем токен из docker-compose окружения
HF_TOKEN = os.getenv("HF_TOKEN", "ВАШ_HF_ТОКЕН") 

# АВТООПРЕДЕЛЕНИЕ РОДИТЕЛЬСКОГО КЛАССА:
# Защищает от багов разных версий grpcio-tools (ищет Servicer или Service)
if hasattr(diarization_pb2_grpc, "DiarizationServiceServicer"):
    BaseServicer = diarization_pb2_grpc.DiarizationServiceServicer
elif hasattr(diarization_pb2_grpc, "DiarizationService"):
    BaseServicer = diarization_pb2_grpc.DiarizationService
else:
    raise AttributeError("Не удалось найти базовый gRPC класс в diarization_pb2_grpc.py")

class DiarizationServiceServicer(BaseServicer):
    def __init__(self):
        print("Загрузка ИИ моделей в память...")
        self.model = whisperx.load_model("small", DEVICE, compute_type=COMPUTE_TYPE, language="ru")
        self.model_a, self.metadata = whisperx.load_align_model(language_code="ru", device=DEVICE)
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        
        # Модель для создания векторов голоса (эмбеддингов)
        self.voice_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": DEVICE}
        )
        print("Все модели загружены. gRPC сервер готов.")

    def _extract_voice_vector(self, audio_path, start, end):
        """Извлекает 192-мерный вектор из конкретного участка аудио"""
        try:
            signal, fs = torchaudio.load(audio_path)
            start_frame = int(start * fs)
            end_frame = int(end * fs)
            segment_signal = signal[:, start_frame:end_frame]
            
            if segment_signal.shape[1] < 1600: # если реплика слишком короткая (менеше ~0.1 сек)
                return []
                
            with torch.no_grad():
                emb = self.voice_encoder.encode_batch(segment_signal)
                return emb.flatten().cpu().numpy().tolist()
        except Exception as e:
            print(f"Ошибка извлечения вектора: {e}")
            return []

    def ProcessAudio(self, request_iterator, context):
        # Собираем байты из стрима от Go в один буфер памяти
        audio_buffer = io.BytesIO()
        for chunk in request_iterator:
            audio_buffer.write(chunk.data)
        
        audio_buffer.seek(0)
        temp_filename = "temp_grpc_audio.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_buffer.read())

        try:
            # 1. Обработка ИИ
            audio = whisperx.load_audio(temp_filename)
            result = self.model.transcribe(audio, batch_size=4)
            result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, DEVICE, return_char_alignments=False)
            diarize_segments = self.diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # 2. Формирование ответа Protobuf с эмбеддингами
            proto_segments = []
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                
                # Извлекаем вектор для этого таймкода
                vector = self._extract_voice_vector(temp_filename, start, end)

                proto_segments.append(diarization_pb2.Segment(
                    speaker=segment.get("speaker", "UNKNOWN"),
                    start=round(start, 2),
                    end=round(end, 2),
                    text=segment["text"].strip(),
                    embedding=vector # передаем массив float в Go
                ))

            gc.collect()
            return diarization_pb2.DiarizationResponse(status="success", segments=proto_segments)

        except Exception as e:
            print(f"Ошибка при обработке аудио: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return diarization_pb2.DiarizationResponse(status="error")
        
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    
    # Автоматически находим функцию регистрации сервиса в сервере
    register_func = None
    for attr in dir(diarization_pb2_grpc):
        if attr.startswith("add_") and attr.endswith("_to_server"):
            register_func = getattr(diarization_pb2_grpc, attr)
            break
            
    if register_func is None:
        raise AttributeError("Не найдена функция add_*_to_server в diarization_pb2_grpc.py")
        
    register_func(DiarizationServiceServicer(), server)
    
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC сервер запущен на порту 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()