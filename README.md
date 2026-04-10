# Audio Insights & Entity Extractor 🎙️🔍

> **Academic Project:** This application was originally developed as a homework project for the _Intro to Data Science and Engineering_ course at **Istanbul Technical University (ITU)** in 2024.

A web application that automatically transcribes meeting audio files and extracts key Named Entities (Persons, Organizations, and Locations) from the transcribed text using state-of-the-art NLP models.

Built as part of a robust Python data pipeline combining Audio Speech Recognition (ASR) and Named Entity Recognition (NER), visualized with a dynamic Streamlit frontend.

## 🚀 Features

- **Automatic Speech Recognition (ASR)**: Uses OpenAI's `whisper-tiny` model to accurately transcribe spoken audio to text.
- **Named Entity Recognition (NER)**: Implements `dslim/bert-base-NER` to detect and extract Persons, Organizations, and Locations from the transcription.
- **High Performance**: Optimized with caching mechanisms so that heavy transformer models are only loaded into system memory once.
- **Modern UI**: An intuitive, column-based dashboard built entirely in Streamlit.
- **Local Execution**: Runs entirely on your local hardware securely, no cloud API keys required.

## 🛠️ Technologies Used

- **Python 3**
- **Streamlit**: For the interactive web interface.
- **Hugging Face Transformers**: To serve the deep learning models text and audio processing pipelines.
- **PyTorch**: Used under-the-hood by Hugging Face to evaluate the neural network graphs.
- **FFmpeg**: Required as a system dependency to process audio arrays.

## 💻 Installation

First, ensure you have python installed. It is recommended to use a virtual environment.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/semihcellk/Meeting-Transcriber-NER.git
   cd Meeting-Transcriber-NER
   ```

2. **Install system-level dependencies:**
   You must have `ffmpeg` installed on your system for audio processing. (See `packages.txt`)
   - **Windows:** Use `winget install ffmpeg` or download binaries.
   - **macOS:** `brew install ffmpeg`
   - **Linux:** `sudo apt install ffmpeg`

3. **Install python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎥 Usage

1. Launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```
2. Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
3. Upload a `/wav` audio file using the provided file uploader.
   _(Note: You can use the included `business_dialog.wav` to test it right away)._
4. Wait for the ASR model to complete transcribing, and view the intelligently grouped entities!

---

_Created by [Semih Çelik](https://github.com/semihcellk)_
