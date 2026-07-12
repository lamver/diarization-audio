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
HF_TOKEN = os.getenv("HF_TOKEN", "ВАШ_HF_ТОКЕН") 

# ЧИСТЫЙ КЛАСС БЕЗ НАСЛЕДОВАНИЯ
# Это защищает от любых изменений в библиотеке grpcio-tools
class DiarizationServiceServicer:
    def __init__(self):
        print("Загрузка ИИ моделей в память...")
        self.model = whisperx.load_model("small", DEVICE, compute_type=COMPUTE_TYPE, language="ru")
        self.model_a, self.metadata = whisperx.load_align_model(language_code="ru", device=DEVICE)
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        
        self.voice_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": DEVICE}
        )
        print("Все модели загружены. gRPC сервер готов.")

    def _extract_voice_vector(self, audio_path, start, end):
        try:
            signal, fs = torchaudio.load(audio_path)
            start_frame = int(start * fs)
            end_frame = int(end * fs)
            segment_signal = signal[:, start_frame:end_frame]
            
            if segment_signal.shape < 1600:
                return []
                
            with torch.no_grad():
                emb = self.voice_encoder.encode_batch(segment_signal)
                return emb.flatten().cpu().numpy().tolist()
        except Exception as e:
            print(f"Ошибка извлечения вектора: {e}")
            return []

    def ProcessAudio(self, request_iterator, context):
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

            # 2. Формирование ответа с эмбеддингами
            proto_segments = []
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                
                vector = self._extract_voice_vector(temp_filename, start, end)

                proto_segments.append(diarization_pb2.Segment(
                    speaker=segment.get("speaker", "UNKNOWN"),
                    start=round(start, 2),
                    end=round(end, 2),
                    text=segment["text"].strip(),
                    embedding=vector
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
    
    # Прямая регистрация без перебора атрибутов.
    # В gRPC v2 функция регистрации лежит либо в grpc модуле сервиса, либо в pb2.
    # Проверяем оба варианта напрямую:
    if hasattr(diarization_pb2_grpc, "add_DiarizationServiceServicer_to_server"):
        diarization_pb2_grpc.add_DiarizationServiceServicer_to_server(DiarizationServiceServicer(), server)
    elif hasattr(diarization_pb2, "add_DiarizationServiceServicer_to_server"):
        diarization_pb2.add_DiarizationServiceServicer_to_server(DiarizationServiceServicer(), server)
    elif hasattr(diarization_pb2_grpc, "add_DiarizationService_to_server"):
        diarization_pb2_grpc.add_DiarizationService_to_server(DiarizationServiceServicer(), server)
    else:
        # Если компилятор совсем всё переиначил, gRPC предоставляет универсальный fallback метод:
        try:
            diarization_pb2_grpc.DiarizationService.RegisterService(DiarizationServiceServicer(), server)
        except Exception:
            raise AttributeError(
                "gRPC не смог автоматически зарегистрировать сервис. "
                "Проверьте, что в diarization.proto имя сервиса указано как DiarizationService"
            )
        
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC сервер запущен на порту 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()