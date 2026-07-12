import grpc
from concurrent import futures
import io
import os
import gc
import whisperx

import diarization_pb2
import diarization_pb2_grpc

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
HF_TOKEN = "ВАШ_HF_ТОКЕН"

class DiarizationServiceServicer(diarization_pb2_grpc.DiarizationServiceServicer):
    def __init__(self):
        print("Загрузка ИИ моделей в память...")
        self.model = whisperx.load_model("small", DEVICE, compute_type=COMPUTE_TYPE, language="ru")
        self.model_a, self.metadata = whisperx.load_align_model(language_code="ru", device=DEVICE)
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("Модели загружены. gRPC сервер готов.")

    def ProcessAudio(self, request_iterator, context):
        # Собираем байты из стрима от Go в один буфер памяти
        audio_buffer = io.BytesIO()
        for chunk in request_iterator:
            audio_buffer.write(chunk.data)
        
        audio_buffer.seek(0)
        
        # WhisperX требует путь к файлу, поэтому сохраняем буфер во временный файл памяти
        # (Либо в /tmp/ на Linux для максимальной скорости)
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

            # 2. Формирование ответа Protobuf
            proto_segments = []
            for segment in result["segments"]:
                proto_segments.append(diarization_pb2.Segment(
                    speaker=segment.get("speaker", "UNKNOWN"),
                    start=round(segment["start"], 2),
                    end=round(segment["end"], 2),
                    text=segment["text"].strip()
                ))

            gc.collect()
            return diarization_pb2.DiarizationResponse(status="success", segments=proto_segments)

        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return diarization_pb2.DiarizationResponse(status="error")
        
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    diarization_pb2_grpc.add_DiarizationServiceServicer_to_server(DiarizationServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC сервер запущен на порту 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()