import os
import subprocess
import threading
import time
from queue import Queue

import google.generativeai as genai
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydantic import BaseModel

from VoiceActivityDetection import VADDetector

load_dotenv()

AUDIO_DIR = "audio"
PIPER_DIR = "piper"
PIPER_BIN = os.path.join(PIPER_DIR, "piper.exe")
PIPER_MODEL = os.path.join(PIPER_DIR, "en_GB-alan-medium.onnx")
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

master_prompt = (
    "You are Jarvis, a helpful AI assistant. "
    "Always address the user as Sir. "
    "Respond briefly and clearly (maximum 3 sentences). "
    "Focus on one task at a time and avoid unnecessary details. "
    "Do not give dangerous instructions or attempt system modifications beyond allowed commands. "
    "If user provides context, use it directly without repeating it back as memory. "
    "Do not invent or store changeable data like weather, stock prices, or time. "
    "Always act like a professional assistant, efficient and reliable."
)


class ChatMLMessage(BaseModel):
    role: str
    content: str


class Client:
    def __init__(self, startListening=True):
        self.greet()
        self.listening = False
        self.history = []
        self.vad_data = Queue()
        self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        self.vad = VADDetector(lambda: None, self.onSpeechEnd, sensitivity=0.5)

        if startListening:
            self.toggleListening()
            self.startListening()
            t = threading.Thread(target=self.transcription_loop, daemon=True)
            t.start()

    def greet(self):
        print("\n\033[36mWelcome to Jarvis\033[0m\n")

    def startListening(self):
        t = threading.Thread(target=self.vad.startListening, daemon=True)
        t.start()

    def toggleListening(self):
        if not self.listening:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "beep.wav"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("\033[36mListening...\033[0m")

        while not self.vad_data.empty():
            self.vad_data.get()

        self.listening = not self.listening

    def onSpeechEnd(self, data):
        if data.any():
            self.vad_data.put(data)

    def addToHistory(self, content: str, role: str):
        if role == "user":
            print(f"\033[37mYou: {content}\033[0m")
            content = f"""{master_prompt}\n\n{content}"""
        else:
            print(f"\033[36mJarvis: {content}\033[0m")
        self.history.append(ChatMLMessage(content=content, role=role))

    def getHistoryAsString(self):
        return "".join(f"<|{msg.role}|>{msg.content}<|end|>\n" for msg in self.history)

    def transcription_loop(self):
        while True:
            if not self.vad_data.empty():
                data = self.vad_data.get()

                if self.listening and len(data) > 12000:
                    self.toggleListening()

                    filename = os.path.join(AUDIO_DIR, "speech.wav")
                    sf.write(filename, data, 16000)

                    segments, info = self.whisper_model.transcribe(
                        filename, language="en"
                    )
                    text = " ".join([seg.text for seg in segments]).strip()

                    if not text:
                        self.toggleListening()
                        continue

                    self.addToHistory(text, "user")
                    history = self.getHistoryAsString()
                    response = gemini_model.generate_content(
                        history + "\n<|assistant|>"
                    )
                    reply = response.text.strip()
                    self.addToHistory(reply, "assistant")
                    self.speak(reply)

    def speak(self, text):
        output_file = os.path.join(AUDIO_DIR, "output.wav")
        process = subprocess.Popen(
            [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", output_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        process.communicate(input=text)

        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", output_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)

        self.toggleListening()


if __name__ == "__main__":
    jc = Client(startListening=True)
    while True:
        time.sleep(1)
