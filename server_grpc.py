import grpc
from concurrent import futures
import io
import os
import gc
import uuid
import whisperx
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# 1. ИМПОРТИРУЕМ НАПРЯМУЮ ИЗ МОДУЛЯ DIARIZE
from whisperx.diarize import DiarizationPipeline  

import diarization_pb2
import diarization_pb2_grpc

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
HF_TOKEN = os.getenv("HF_TOKEN", "") 

class DiarizationServiceServicer(diarization_pb2_grpc.DiarizationServiceServicer):
    def __init__(self):
        print("Загрузка ИИ моделей в память...")
        self.model = whisperx.load_model("small", DEVICE, compute_type=COMPUTE_TYPE, language="ru")
        self.model_a, self.metadata = whisperx.load_align_model(language_code="ru", device=DEVICE)
        
        # 2. ИСПРАВЛЕНА ИНИЦИАЛИЗАЦИЯ (аргумент token вместо use_auth_token)
        self.diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
        
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
        temp_filename = f"temp_grpc_{uuid.uuid4()}.wav"
        try:
            with io.BytesIO() as audio_buffer:
                for chunk in request_iterator:
                    audio_buffer.write(chunk.data)
                audio_buffer.seek(0)
                with open(temp_filename, "wb") as f:
                    f.write(audio_buffer.read())

            audio = whisperx.load_audio(temp_filename)
            result = self.model.transcribe(audio, batch_size=4)
            result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, DEVICE, return_char_alignments=False)
            
            # Работает со встроенным пайплайном
            diarize_segments = self.diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            proto_segments = []
            for segment in result["segments"]:
                start = segment.get("start", 0.0)
                end = segment.get("end", 0.0)
                
                vector = self._extract_voice_vector(temp_filename, start, end)

                proto_segments.append(diarization_pb2.Segment(
                    speaker=segment.get("speaker", "UNKNOWN"),
                    start=round(start, 2),
                    end=round(end, 2),
                    text=segment.get("text", "").strip(),
                    embedding=vector
                ))

            return diarization_pb2.DiarizationResponse(status="success", segments=proto_segments)

        except Exception as e:
            print(f"Ошибка при обработке аудио: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return diarization_pb2.DiarizationResponse(status="error")
        
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    print(f"Не удалось удалить {temp_filename}: {e}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    diarization_pb2_grpc.add_DiarizationServiceServicer_to_server(DiarizationServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC Diarization Server started on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()