import gradio as gr
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import tempfile
import base64
# ------------------- Load Models -------------------
# Load Whisper model and processor for speech-to-text
stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
stt_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Load sentence transformer model for intent classification
intent_encoder = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------- Intent Database -------------------
# Mapping of predefined intents to example sentences
INTENT_DB = {
    "Shopping": ["buy a phone", "shop online", "order groceries", "purchase shoes", "need a new laptop", "buy clothes"],
    "Weather": ["weather today", "will it rain", "temperature now", "weather in Delhi tomorrow", "is it sunny"],
    "YouTube": ["play music", "watch comedy video", "open YouTube", "play cooking tutorial", "play workout mix"],
    "Translate": ["translate hello to French", "how to say good night in German", "translate this to Hindi", "translate to Spanish"],
    "Location": ["show me my live location", "where am I", "find my location", "my current place"],
    "Restaurants": ["restaurants near me", "good food nearby", "where can I eat", "pizza places near me"],
    "Maps": ["directions to airport", "how to reach nearest metro", "navigate to my home", "route to school"],
    "Time": ["what time is it", "current time", "time in New York", "time now"],
    "News": ["latest news", "what's happening today", "breaking news", "headlines now"],
    "Quotes": ["motivate me", "give me an inspirational quote", "quote of the day"],
    "Introduction": ["my name is Alex", "I am John", "hi, I’m Sarah", "this is Michael", "hello, I’m Jane", "I go by Sam"],
    "Greetings": ["hello", "hi", "hey there", "good morning", "good evening"],
    "Wellbeing": ["how are you", "how's it going", "how do you feel", "what's up"],
    "Gratitude": ["thank you", "thanks a lot", "appreciate it", "thank you very much"],
    "Farewell": ["bye", "goodbye", "see you later", "talk to you soon", "catch you later"],
    "Search": ["search for electric cars", "look up best smartphones", "find open source projects"]
}


# ------------------- Action Dispatcher -------------------
# Maps intents to corresponding URLs or actions
ACTION_DISPATCH = {
    "Shopping": lambda q: f"https://www.amazon.in/s?k={q}",
    "Wikipedia": lambda q: f"https://en.wikipedia.org/wiki/{q.replace(' ', '_')}",
    "YouTube": lambda q: f"https://www.youtube.com/results?search_query={q}",
    "Translate": lambda q: f"https://translate.google.com/?sl=auto&tl=en&text={q}",
    "Location": lambda q: "https://www.google.com/maps/search/where+am+I/",
    "Restaurants": lambda q: "https://www.google.com/maps/search/restaurants+near+me/",
    "Weather": lambda q: f"https://www.google.com/search?q=weather+{q}",
    "Maps": lambda q: f"https://www.google.com/maps/search/{q}",
    "Time": lambda q: "https://time.is/",
    "News": lambda q: f"https://news.google.com/search?q={q}",
    "Quotes": lambda q: "https://www.brainyquote.com/quote_of_the_day",
    "Search": lambda q: f"https://www.google.com/search?q={q}",
    "Introduction": lambda q: None,
    "Greetings": lambda q: None,
    "Wellbeing": lambda q: None,
    "Gratitude": lambda q: None,
    "Farewell": lambda q: None,
}


# ------------------- Helper Functions -------------------
# Extracts the name from self-introduction sentences
def extract_name(text):
    for word in text.split():
        if word.istitle() and word.lower() not in ["i", "am", "my", "name", "is", "this"]:
            return word
    return None

# Provides personalized or static responses for non-redirect intents
def get_personal_response(intent, transcription):
    name = extract_name(transcription) if intent == "Introduction" else None
    responses = {
        "Introduction": f"Nice to meet you, {name}!" if name else "Nice to meet you!",
        "Greetings": "Hello! How can I help you today?",
        "Wellbeing": "I'm just a bunch of code, but I'm here and ready to help!",
        "Gratitude": "You're welcome!",
        "Farewell": "Goodbye! Talk to you later."
    }
    return responses.get(intent, "How can I assist you?")

# Precompute intent embeddings for fast similarity comparison
intent_vectors = {
    intent: intent_encoder.encode(examples, convert_to_tensor=True)
    for intent, examples in INTENT_DB.items()
}

# Finds the best matching intent based on cosine similarity
def get_best_intent(transcription):
    input_vec = intent_encoder.encode(transcription, convert_to_tensor=True)
    best_intent, best_score = None, -1
    for intent, vecs in intent_vectors.items():
        score = util.pytorch_cos_sim(input_vec, vecs).max().item()
        if score > best_score:
            best_intent, best_score = intent, score
    return best_intent

# Converts TTS audio output into base64 string for browser playback
def tts_to_base64(text):
    tts = gTTS(text=text)
    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)
    with open(tts_file.name, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Core function: handles audio input, intent recognition, response generation and redirection
def process_audio(audio_path):
    # Load audio with librosa at 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    # Preprocess for Whisper
    inputs = stt_processor(audio, sampling_rate=16000, return_tensors="pt")

    # Force language to English for decoding
    forced_ids = stt_processor.get_decoder_prompt_ids(language="en", task="transcribe")

    # Perform speech-to-text
    generated_ids = stt_model.generate(
        inputs.input_features,
        forced_decoder_ids=forced_ids
    )
    transcription = stt_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Detect the best intent from the transcription
    best_intent = get_best_intent(transcription)

    # Generate response and determine redirection
    if best_intent:
        if best_intent in ["Introduction", "Greetings", "Wellbeing", "Gratitude", "Farewell"]:
            response = get_personal_response(best_intent, transcription)
            url = ""
        else:
            response = f"Intent: {best_intent}. Taking you there..."
            query = transcription.replace(" ", "+")
            url = ACTION_DISPATCH[best_intent](query)
    else:
        response = "Sorry, I couldn't understand that."
        url = ""

    # Convert text response to base64-encoded audio
    b64_audio = tts_to_base64(response)

    # Prepare HTML for audio playback + optional redirection
    redirect_html = f"""
    <audio autoplay onended=\"window.location.href='{url}'\">
        <source src=\"data:audio/mp3;base64,{b64_audio}\" type=\"audio/mp3\">
    </audio>""" if url else f"""
    <audio autoplay>
        <source src=\"data:audio/mp3;base64,{b64_audio}\" type=\"audio/mp3\">
    </audio>"""

    return transcription, best_intent or "Unknown", response, redirect_html


# ------------------- Gradio Interface -------------------
# Prepare dropdown list and example lookup
intent_options = list(INTENT_DB.keys())
intent_examples = {
    intent: "\n".join(f"- {ex}" for ex in examples)
    for intent, examples in INTENT_DB.items()
}

with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Voice-Based Intent Assistant")
    gr.Markdown("Speak a natural sentence — the assistant will detect your intent and take action accordingly.")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="🎤 Speak Your Command")

    with gr.Row():
        transcription = gr.Text(label="📝 You Said")
        detected_intent = gr.Text(label="📌 Detected Intent")
        response_text = gr.Text(label="💬 Assistant Response")

    result_html = gr.HTML(label="🔊 Voice + Redirection")

    gr.Markdown("---")
    gr.Markdown("### 💡 What Can You Say?")

    intent_dropdown = gr.Dropdown(label="Choose an Intent to See Examples", choices=intent_options)
    example_output = gr.Markdown()

    # Display intent examples when selected
    def show_examples(selected_intent):
        return intent_examples[selected_intent] if selected_intent in intent_examples else "Select an intent to see examples."

    intent_dropdown.change(show_examples, inputs=intent_dropdown, outputs=example_output)

    # Connect the audio input to the core processing function
    audio_input.change(
        fn=process_audio,
        inputs=audio_input,
        outputs=[transcription, detected_intent, response_text, result_html]
    )

# Launch the Gradio app
demo.launch()