import os
import re
import nltk
import unicodedata
from textblob import TextBlob
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

# Explicitly set the correct NLTK data path
nltk.data.path.append("C:\\Users\\Simbyot\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\Roaming\\nltk_data")

# Define paths
source_folder = "source"
prepared_folder = "prepared"

# Ensure output folder exists
os.makedirs(prepared_folder, exist_ok=True)

# Regular expressions for filtering unwanted text
chapter_patterns = [
    r"^\s*Chapter\s+\d+[\.\s]*$",  
    r"^\s*CHAPTER\s+\d+[\.\s]*$",  
    r"^\s*Chapter\s+[IVXLCDM]+[\.\s]*$",  
    r"^\s*CHAPTER\s+[IVXLCDM]+[\.\s]*$",  
    r"^\s*-+\s*$",  
]

volume_patterns = r"^\s*VOLUME\s+[A-Z0-9]+\s*$"
number_m_pattern = r"^\s*\d{4,}m\s*$"
roman_numerals = r"^([IVXLCDM]+)\s*$"
incomplete_sentences = r"^(said\s+[A-Z][a-z]+)\.$"
numbers_with_letters = r"^\d+[a-zA-Z]+\s+.*$"

# Define abbreviations to remove
common_abbreviations = ["M", "Mr", "Mrs", "Dr", "St", "Jr", "Sr", "Prof", "Gen", "e.g", "i.e"]
abbreviation_pattern = re.compile(r"\b(" + "|".join(common_abbreviations) + r")\.\s+([A-Z][a-z]+)")

def normalize_text(text):
    """Converts accented characters to their closest ASCII equivalent (e.g., è → e)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

def fix_missing_spaces(text):
    """Fixes missing spaces between words, especially when a lowercase letter is followed by an uppercase letter."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

def clean_text(text):
    """Cleans the text by removing unwanted characters and structures while preserving readability."""
    
    text = normalize_text(text)
    
    # Remove common abbreviations but keep the name (e.g., "Mr. Smith" → "Smith")
    text = abbreviation_pattern.sub(r"\2", text)

    # Replace em-dashes and multiple dashes with a space
    text = re.sub(r"(?<=\w)—(?=\w)", " — ", text)  
    text = re.sub(r"\s*—\s*", " ", text)  
    text = re.sub(r"\s*--+\s*", " ", text)  

    # Remove underscores
    text = re.sub(r"_", "", text)

    # Remove all types of quotes
    text = re.sub(r"[“”‘’\"]", "", text)

    # Ensure proper spacing after punctuation
    text = re.sub(r"([?!])(\S)", r"\1 \2", text)

    # Remove all punctuation except . , ? !
    text = re.sub(r"[^\w\s.,?!]", "", text)

    # Fix missing spaces (e.g., "MarseillesThe Arrival" → "Marseilles The Arrival")
    text = fix_missing_spaces(text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"^,\s*", "", sentence)

        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:] if sentence else ""

        # Skip unwanted text patterns
        if any(re.match(pattern, sentence, re.IGNORECASE) for pattern in chapter_patterns):
            continue
        if re.match(volume_patterns, sentence, re.IGNORECASE):
            continue
        if re.match(number_m_pattern, sentence):
            continue
        if re.match(roman_numerals, sentence):
            continue
        if re.match(incomplete_sentences, sentence):
            continue
        if re.match(numbers_with_letters, sentence):
            continue

        if sentence:
            cleaned_sentences.append(sentence)

    return cleaned_sentences

def process_text_with_textblob(sentences):
    """Uses TextBlob to further refine text by removing duplicates and incomplete sentences."""
    processed_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        blob = TextBlob(sentence)

        # Remove if duplicate
        if sentence in seen_sentences:
            continue

        # Remove if sentence is too short (likely incomplete)
        if len(sentence.split()) < 3:
            continue
        
        # Remove if not a complete sentence
        if not sentence.endswith(('.', '?', '!')):
            continue

        seen_sentences.add(sentence)
        processed_sentences.append(sentence)

    return processed_sentences

def should_process_file(input_path, output_path):
    """Checks if the file should be processed based on timestamps."""
    if not os.path.exists(output_path):
        return True  # File doesn't exist, needs processing
    
    # Compare modification times
    source_mod_time = os.path.getmtime(input_path)
    prepared_mod_time = os.path.getmtime(output_path)

    return source_mod_time > prepared_mod_time  # Process only if source is newer

def process_files():
    """Processes only new or modified .txt files in the source folder."""
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(source_folder, filename)
            output_path = os.path.join(prepared_folder, filename)

            if not should_process_file(input_path, output_path):
                print(f"Skipping (already processed): {filename}")
                continue  # Skip already processed files

            with open(input_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Step 1: Initial text cleaning
            cleaned_sentences = clean_text(text)

            # Step 2: Further processing with TextBlob
            final_sentences = process_text_with_textblob(cleaned_sentences)

            # Ensure each sentence is on a new line
            final_text = "\n".join(final_sentences)

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(final_text)

            print(f"Processed: {filename}")

if __name__ == "__main__":
    process_files()