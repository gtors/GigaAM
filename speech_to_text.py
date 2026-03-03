import collections
import os
import sys
import tempfile
import wave
import pyaudio
import webrtcvad
import onnx_asr 

# ------------------ Настройки ------------------
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 для 30 мс
PADDING_DURATION_MS = 400  # сколько тишины считать концом фразы
NUM_PADDING_FRAMES = int(PADDING_DURATION_MS / FRAME_DURATION_MS)

# VAD aggressiveness: 0 — самый мягкий, 3 — самый строгий
vad = webrtcvad.Vad(3)

# Загружаем модель один раз при старте (лучше всего на CPU)
# Варианты для GigaAM v3 (2026 год):
#   "gigaam-v3-ctc"          — быстрее, чуть хуже пунктуация
#   "gigaam-v3-rnnt"         — медленнее, но лучше качество
#   "gigaam-v3-e2e-rnnt"     — end-to-end RNNT с пунктуацией и нормализацией 
print("Загружаю модель GigaAM v3 (ONNX) ... может занять 10–40 секунд при первом запуске")
model = onnx_asr.load_model("gigaam-v3-ctc")
# Альтернативы:
# model = onnx_asr.load_model("gigaam-v3-ctc")
# model = onnx_asr.load_model("gigaam-v3-rnnt")

print("Модель загружена. Говорите в микрофон...")
# ------------------------------------------------

pa = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=FRAME_SIZE
)

# Кольцевой буфер для детекции начала речи
ring_buffer = collections.deque(maxlen=NUM_PADDING_FRAMES)
speech_buffer = []          # текущая фраза
triggered = False           # находимся внутри фразы?

try:
    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        ring_buffer.append((frame, is_speech))

        if not triggered:
            # Проверяем, началась ли речь
            voiced_count = sum(1 for _, speech in ring_buffer if speech)
            if voiced_count > 0.7 * len(ring_buffer):  # порог начала
                triggered = True
                # Переносим накопленные кадры в речь
                speech_buffer.extend(f for f, _ in ring_buffer)
                ring_buffer.clear()
        else:
            speech_buffer.append(frame)

            # Проверяем, закончилась ли речь
            unvoiced_count = sum(1 for _, speech in ring_buffer if not speech)
            if unvoiced_count > 0.8 * len(ring_buffer):  # порог конца
                triggered = False

                # Сохраняем фразу во временный wav
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    wf = wave.open(tmp.name, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b''.join(speech_buffer))
                    wf.close()

                    # Распознаём
                    try:
                        text = model.recognize(tmp.name)
                        if text.strip():
                            print("→", text)
                            sys.stdout.flush()
                    except Exception as e:
                        print("Ошибка распознавания:", e)

                # Уборка
                os.unlink(tmp.name)
                speech_buffer = []
                ring_buffer.clear()

except KeyboardInterrupt:
    print("\nОстановка...")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
