
import streamlit as st
from transformers import pipeline, AutoModelForSpeechRecognition, AutoTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer as AutoSeq2SeqTokenizer
import pyaudio
import numpy as np
import torch
import concurrent.futures
import base64
import cv2
from PIL import Image

class AI_Assistant:
    def __init__(self):
        self.speech_recognition_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-small-960h")
        self.text_to_speech_pipeline = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech-small")
        self.conversation_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        self.conversation_tokenizer = AutoSeq2SeqTokenizer.from_pretrained("t5-small")
        self.multimodal_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        self.multimodal_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.full_transcript = [
            {"role": "system", "content": "You are a language model called Llama 3 created by Meta, answer the questions being asked in less than 300 characters. Do not bold or asterix anything because this will be passed to a text to speech service."},
        ]

    def start_transcription(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        while True:
            audio_data = stream.read(1024)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                transcript = executor.submit(self.speech_recognition_pipeline, audio_data).result()
                if transcript:
                    self.generate_ai_response(transcript["text"])
                    st.write(transcript["text"])

    def generate_ai_response(self, transcript):
        self.full_transcript.append({"role": "user", "content": transcript})
        st.write(f"\nUser:{transcript}")
        input_ids = self.conversation_tokenizer.encode(" ".join([turn["content"] for turn in self.full_transcript]), return_tensors="pt")
        output = self.conversation_model.generate(input_ids, max_length=512)
        response = self.conversation_tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("Llama 3:", response)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            audio = executor.submit(self.text_to_speech_pipeline, response).result()
        st.write(response)
        # Play the audio using JavaScript
        audio_data = audio.audio.numpy().tobytes()
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        st.markdown(f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
        """, unsafe_allow_html=True)
        self.full_transcript.append({"role": "assistant", "content": response})

    def process_webcam_feed(self, frame):
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use the multimodal model to generate a response
        inputs = self.multimodal_tokenizer("Describe the image", return_tensors="pt")
        outputs = self.multimodal_model.generate(**inputs)
        response = self.multimodal_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def process_uploaded_file(self, file):
        # Use the multimodal model to generate a response
        inputs = self.multimodal_tokenizer("Describe the image", return_tensors="pt")
        outputs = self.multimodal_model.generate(**inputs)
        response = self.multimodal_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    st.title("AI Assistant")
    ai_assistant = AI_Assistant()
    webcam_feed = st.camera_input("Webcam")

    if webcam_feed:
        frame = webcam_feed.get_frame()
        response = ai_assistant.process_webcam_feed(frame)
        st.write(response)

    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file:
        image = Image.open(uploaded_file)
        response = ai_assistant.process_uploaded_file(image)
        st.write(response)

    ai_assistant.start_transcription()

if __name__ == "__main__":
    main() 
