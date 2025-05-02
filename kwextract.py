import os
# from moviepy.editor import VideoFileClip
import whisper
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from hindi_stopwords import stopwords
import ffmpeg


def extract_audio(video_path, audio_path="temp_audio.wav"):
    ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True)
    return audio_path


def transcribe_audio(audio_path, model_size='base'):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language='hi')
    return result['text']


import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

def clean_and_count_words(text):
    # Tokenize text
    words = word_tokenize(text)

    # Define stopwords
    stopwords_en = set(stopwords.words("english"))
    try:
        stopwords_hi = set(stopwords.words("hindi"))
    except:
        stopwords_hi = set()  # Fallback if Hindi stopwords aren't available

    # Clean and filter
    cleaned_words = []
    for word in words:
        word = word.lower()
        word = re.sub(r"[^\w]", "", word)  # Remove punctuation

        if word.isalpha() and word not in stopwords_en and word not in stopwords_hi:
            cleaned_words.append(word)

    # Count word frequencies
    return dict(Counter(cleaned_words))




def save_output(transcription, word_freq):
    # Write transcription to file
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(transcription)

    # Sort word frequencies in descending order
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Write sorted word frequencies to file
    with open("word_frequencies.txt", "w", encoding="utf-8") as f:
        for word, freq in sorted_word_freq:
            f.write(f"{word}: {freq}\n")


def process_video(video_path):
    audio_path = extract_audio(video_path)
    transcription = transcribe_audio(audio_path)
    word_freq = clean_and_count_words(transcription)
    save_output(transcription, word_freq)
    os.remove(audio_path)
    print("Transcription and keyword extraction completed.")


process_video("[Video__Path]")
