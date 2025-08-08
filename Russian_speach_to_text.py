# pip install openai
# pip install openai-whisper
# pip install -U openai-whisper
import openai
openai.api_key = "sk-proj-xxxxx"
# import libraries
import openai
from openai import OpenAI
import whisper

# import warnings
# import textwrap

# ignore any python warnings
# warnings.filterwarnings("ignore")

# pick the desired model from the predifined available models in the library
model = whisper.load_model("base")
#audio_path = r"C:\Endava\EndevLocal\AI_speech_text_summarize\russian2.mp3"
#audio_path = r"C:\Users\ocramar\Downloads\russian4.mp3"
audio_path = r"C:\Users\ocramar\Downloads\russian59.ogg"
# get the transcript of the audio file
result = model.transcribe(audio_path, fp16=False, language="ru")

# specify the transcript
transcript = result["text"]

# specify the file name
file_path = "russian50.txt"

# save the transcript to the file
with open(file_path, "w", encoding="utf-8") as file: file.write(transcript)

# display the result
# print(textwrap.fill(result["text"], 191))
print(result["text"])

# Translate the Russian transcript to English using OpenAI API
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-ru-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

translated = translation_model.generate(**tokenizer(transcript, return_tensors="pt", padding=True))
english_translation = tokenizer.decode(translated[0], skip_special_tokens=True)

print("\n🌍 English Translation:\n", english_translation)

with open("english.txt", "w", encoding="utf-8") as f:
    f.write(english_translation)
