import collections

import numpy as np
import pyaudio
import webrtcvad

RATE = 16000
CHUNK = 160  # 10ms @16kHz
CHANNELS = 1
FORMAT = pyaudio.paInt16

audio = pyaudio.PyAudio()


class VADDetector:
    def __init__(self, onSpeechStart, onSpeechEnd, sensitivity=0.4):
        """
        onSpeechStart: callback when speech starts
        onSpeechEnd: callback when speech ends (returns numpy array int16)
        sensitivity: seconds of silence before considering speech ended
        """
        self.sample_rate = RATE
        self.vad = webrtcvad.Vad(3)  # 0-3, 3 = most aggressive
        self.frameHistory = collections.deque(maxlen=50)
        self.block_since_last_spoke = 0
        self.onSpeechStart = onSpeechStart
        self.onSpeechEnd = onSpeechEnd
        self.voiced_frames = collections.deque(maxlen=2000)
        self.silence_blocks = int(sensitivity * 100)
        self.is_speaking = False  # state flag

    def voice_activity_detection(self, audio_data: bytes):
        return self.vad.is_speech(audio_data, self.sample_rate)

    def audio_callback(self, audio_data: bytes):
        detection = self.voice_activity_detection(audio_data)

        if not self.is_speaking and detection:
            self.is_speaking = True
            self.onSpeechStart()
            self.voiced_frames.append(audio_data)
            self.block_since_last_spoke = 0

        elif self.is_speaking and detection:
            self.voiced_frames.append(audio_data)
            self.block_since_last_spoke = 0

        elif self.is_speaking and not detection:
            if self.block_since_last_spoke >= self.silence_blocks:
                if len(self.voiced_frames) > 0:
                    samp = b"".join(self.voiced_frames)
                    self.onSpeechEnd(np.frombuffer(samp, dtype=np.int16))
                self.voiced_frames.clear()
                self.is_speaking = False
                self.block_since_last_spoke = 0
            else:
                self.block_since_last_spoke += 1

        self.frameHistory.append(detection)

    def startListening(self):
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_callback(data)
        except KeyboardInterrupt:
            print("Stopped by user.")
        except Exception as e:
            print("Error:", e)
        finally:
            stream.stop_stream()
            stream.close()


if __name__ == "__main__":

    def onSpeechStart():
        print("Speech started...")

    def onSpeechEnd(data):
        print("Speech ended. Length:", len(data))

    vad = VADDetector(onSpeechStart, onSpeechEnd, sensitivity=0.5)
    vad.startListening()
