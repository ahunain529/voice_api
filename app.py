from flask import Flask, request, jsonify, send_file
from gtts import gTTS
import os
import re
import google.generativeai as genai

app = Flask(__name__)

# Gemini setup
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat(history=[])

def clean_text(text):
    text = re.sub(r"[*_#>`~\-]+", '', text)
    text = re.sub(r"\n+", '. ', text)
    return text.strip()

@app.route('/text-to-ai', methods=['POST'])
def text_to_ai():
    user_text = request.json.get('text', '')
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    full_prompt = (
        "Answer briefly and to the point. "
        "Use simple, clear language.\n"
        f"User: {user_text}\nAI:"
    )
    response = chat.send_message(full_prompt)
    reply = clean_text(response.text)

    # Generate voice response
    tts = gTTS(reply, lang='en', slow=False)
    tts.save("response.mp3")

    return jsonify({"reply": reply, "audio_url": request.host_url + "get-audio"})

@app.route('/get-audio', methods=['GET'])
def get_audio():
    return send_file("response.mp3", mimetype="audio/mpeg")

if __name__ == '__main__':
    app.run(debug=True)
