# 🎙️ Gradio-Powered Voice Assistant

An interactive, browser-based voice assistant that listens to your audio, understands what you're saying using AI models, and responds accordingly — either with a spoken response or by opening a relevant web page.

---

## 🔥 Features

- 🗣️ **Speech-to-Text with Whisper** – Converts your spoken commands into text.
- 🧠 **Intent Detection with Sentence Transformers** – Understands your intent using semantic similarity.
- 🌍 **Smart Actions** – Opens YouTube, Google Maps, Weather, Amazon, Translate, and more.
- 💬 **Natural Voice Responses** – Replies with spoken feedback using gTTS.
- 🌐 **Gradio UI** – Clean web interface, no need to install any GUI.
- ⚡ **Fast Processing** – Lightweight, efficient, and responsive.

---

## 📷 Preview

_Example: Say “play music on YouTube” → Assistant detects “YouTube” intent → Opens YouTube search with your query._

---

## 🛠️ Tech Stack

- [OpenAI Whisper](https://github.com/openai/whisper) – Speech-to-text
- [Sentence Transformers](https://www.sbert.net/) – Intent recognition
- [gTTS (Google Text-to-Speech)](https://pypi.org/project/gTTS/) – Spoken responses
- [Gradio](https://www.gradio.app/) – Web interface for AI apps
- `librosa` – Audio preprocessing

---

## ⚙️ Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/gradio-voice-assistant.git
cd gradio-voice-assistant
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Run the App

bash
Copy code
python app.py
Visit in Your Browser
Once running, it will open automatically at:

arduino
Copy code
http://localhost:7860
🧠 Supported Intents
You can say things like:

“Play workout music” → 🎵 YouTube

“What's the weather in Delhi?” → 🌦️ Weather

“Translate hello to French” → 🌐 Google Translate

“Find my location” → 📍 Google Maps

“Motivate me” → 🧘 Inspirational quote

“Buy a phone” → 🛍️ Amazon Shopping

You can also explore intent examples from the dropdown in the UI.

🧩 Add New Intents
To extend functionality:

Add new phrases in INTENT_DB.

Add a matching URL action in ACTION_DISPATCH.

📄 License
MIT License – Free to use, modify, and share.

🙌 Acknowledgements
OpenAI for Whisper

SBERT Team for Sentence Transformers

Gradio for the awesome UI framework

yaml
Copy code
