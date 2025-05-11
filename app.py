from flask import Flask, request, jsonify
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import re
import google.generativeai as genai
import os
import uuid

app = Flask(__name__)

# Google Gemini Setup
GOOGLE_API_KEY = "AIzaSyCJeBrJ0liMxye8rEgScMfUqjv7mLEoRhQ"
genai.configure(api_key=GOOGLE_API_KEY)

# Load Whisper and Gemini
whisper_model = whisper.load_model("tiny")
model = genai.GenerativeModel('gemini-1.5-flash')

# Utilities
def clean_text_for_speech(text):
    text = re.sub(r"[*_#>`~\-]+", '', text)
    text = re.sub(r"\n+", '. ', text)
    return text.strip()

def record_audio(user_id, duration=4):
    fs = 16000
    filename = f"{user_id}.wav"
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    audio = np.squeeze(recording)
    scipy.io.wavfile.write(filename, fs, audio)
    return filename

def transcribe_audio(filename):
    result = whisper_model.transcribe(filename, fp16=False)
    return result["text"].strip()

def get_gemini_reply(prompt):
    instruction = (
        "Answer briefly and to the point. Focus only on what's asked. "
        "Avoid extra explanation unless necessary. Respond in clear, simple language."
    )
    full_prompt = f"{instruction}\nUser: {prompt}\nAI:"
    chat = model.start_chat(history=[])
    response = chat.send_message(full_prompt)
    return clean_text_for_speech(response.text)

# Routes
@app.route('/api/text', methods=['POST'])
def text_input():
    data = request.get_json()
    user_input = data.get('text', '')
    reply = get_gemini_reply(user_input)
    return jsonify({'reply': reply})

@app.route('/api/voice', methods=['POST'])
def voice_input():
    user_id = str(uuid.uuid4())
    try:
        filename = record_audio(user_id)
        user_text = transcribe_audio(filename)
        reply = get_gemini_reply(user_text)
        return jsonify({'transcription': user_text, 'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(f"{user_id}.wav"):
            os.remove(f"{user_id}.wav")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
