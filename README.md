# Jarvis (Minimal Voice Assistant)

A minimal, local-first voice assistant that listens for speech, transcribes it with Whisper (via **faster-whisper**), generates a short reply with **Gemini**, and speaks with **Piper TTS**. No wake word, no barge-in — just a clean loop optimized for stability.

## Features
- Voice Activity Detection (**WebRTC VAD**) to auto-segment speech  
- Speech-to-Text with **faster-whisper** (small, CPU int8 by default)  
- Short, safe replies from **Gemini** (configurable master prompt)  
- Text-to-Speech with **Piper TTS** + **ffplay** playback  
- Simple console logs and minimal state/history  

---

## Quick Start

### 1) Prerequisites
- Python 3.11+  
- **FFmpeg** (for `ffplay`)  
- **Piper TTS** binary + a voice model (see links below)
  
### 2) Install dependencies
pip install -r requirements.txt

### 3) Configure environment
Create a .env file in project root:
GEMINI_API_KEY=your_api_key_here

### 4) Put Piper files
- Place piper.exe (or piper on Linux/macOS) under ./piper/
- Place a voice model (e.g., en_GB-alan-medium.onnx) under ./piper/

## Piper TTS Downloads
Piper repo & releases (binaries):

- GitHub repository: https://github.com/rhasspy/piper

- Releases (Windows/Linux/macOS builds): https://github.com/rhasspy/piper/releases

Piper voices (models):

- Hugging Face voice collection (many languages): https://huggingface.co/rhasspy/piper-voices

- Voice samples page: https://rhasspy.github.io/piper-samples/

Pick a voice .onnx model and place it under ./piper/. Update PIPER_MODEL in main.py if you change the filename.

## Credits

- Piper TTS by Rhasspy community

- faster-whisper by SYSTRAN

- WebRTC VAD Python bindings
