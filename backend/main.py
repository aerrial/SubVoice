import os
import io
import wave
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from dotenv import load_dotenv
from groq import Groq
import requests

load_dotenv()

app = FastAPI()

# Ініціалізація клієнта Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

# Параметри аудіо з Android (16kHz, mono, 16-bit PCM)
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

def pcm_to_wav(pcm_data: bytes) -> bytes:
    """Конвертує сирі PCM-байти у форматизований WAV файл у пам'яті"""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_data)
    wav_io.seek(0)
    return wav_io.read()

def translate_text_deepl(text: str, target_lang: str = "RU") -> str:
    """Переклад тексту через DeepL API"""
    if not text.strip():
        return ""
    url = "https://api-free.deepl.com/v2/translate"
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    data = {
        "text": [text],
        "target_lang": target_lang
    }
    try:
        response = requests.post(url, headers=headers, data=data)
        result = response.json()
        return result["translations"][0]["text"]
    except Exception as e:
        print(f"Помилка перекладу DeepL: {e}")
        return text

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Клієнт Android підключився до WebSocket")
    
    pcm_buffer = bytearray()
    # Буферизуємо приблизно 3 секунди аудіо перед відправкою на STT
    # 16000 samples/sec * 2 bytes/sample * 3 sec = 96000 bytes
    BUFFER_THRESHOLD = 96000 

    try:
        while True:
            # Отримуємо сирі байти з мікрофона Android
            data = await websocket.receive_bytes()
            pcm_buffer.extend(data)

            # Коли накопичили достатньо аудіо для розпізнавання
            if len(pcm_buffer) >= BUFFER_THRESHOLD:
                chunk_to_process = bytes(pcm_buffer)
                pcm_buffer.clear()

                # 1. Конвертуємо PCM у WAV
                wav_bytes = pcm_to_wav(chunk_to_process)

                # 2. Відправляємо в Groq Whisper API (STT)
                try:
                    transcription = groq_client.audio.transcriptions.create(
                        file=("audio.wav", wav_bytes),
                        model="whisper-large-v3",
                        prompt="Мова спілкування: румунська або казахська."
                    )
                    original_text = transcription.text.strip()

                    if original_text:
                        print(f"🗣️ Розпізнано: {original_text}")

                        # 3. Переклад на російську
                        translated_text = translate_text_deepl(original_text, target_lang="RU")
                        print(f"🌐 Переклад: {translated_text}")

                        # 4. Відправляємо текстову відповідь назад на Android
                        await websocket.send_json({
                            "type": "text_result",
                            "original": original_text,
                            "translation": translated_text
                        })

                except Exception as e:
                    print(f"Помилка розпізнавання/перекладу: {e}")

    except WebSocketDisconnect:
        print("📱 Клієнт Android відключився")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)