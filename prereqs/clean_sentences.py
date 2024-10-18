import json
import re
from collections import defaultdict

# Function to preprocess and clean sentences
def preprocess_sentence(sentence, word):
    # Remove sentences with underscores or slashes
    if '_' in sentence or '/' in sentence:
        return None

    # Remove sentences starting with 'one' or 'two'
    if sentence.lower().startswith('one') or sentence.lower().startswith('two'):
        return None

    # Remove Kannada digits in brackets
    sentence = re.sub(r'\[\d+\]', '', sentence)

    # Remove sentences starting with numbers
    sentence = re.sub(r'^\d+ ', '', sentence)

    # Remove sentences with sequences of numbers and letters
    if re.search(r'\d+[A-Za-z]', sentence):
        return None

    # Remove specific symbols
    for symbol in ['►', '>', '<']:
        if symbol in sentence:
            return None

    # Remove sentences with '...'
    if '...' in sentence:
        return None

    # Remove single-word sentences
    if len(sentence.split()) == 1:
        return None

    # Keep only the first sentence (up to the first period)
    sentence = sentence.split('.')[0] + '.' if '.' in sentence else sentence

    # Remove all types of quotes
    sentence = re.sub(r'[\'"`“”‘’]', '', sentence)

    # General text cleaning
    sentence = re.sub(r'\s+', ' ', sentence)  # Replace multiple spaces with a single space
    sentence = sentence.strip()  # Remove leading and trailing spaces

    # Check if the sentence contains the word, only return if it does
    if word in sentence:
        return sentence
    else:
        return None

# Read JSON file
input_file = 'kannada_sentences_new2.json'
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

cleaned_data = defaultdict(set)  # Use a set to remove duplicates automatically

# Process sentences
for word, sentences in data.items():
    for sentence in sentences:
        cleaned_sentence = preprocess_sentence(sentence, word)
        if cleaned_sentence:
            cleaned_data[word].add(cleaned_sentence)  # Add to set to avoid duplicates

# Convert sets back to lists for JSON serialization
cleaned_data = {word: list(sentences) for word, sentences in cleaned_data.items()}

# Write cleaned data to a new JSON file
output_file_json = 'kannada_sentences_cleaned.json'
with open(output_file_json, 'w', encoding='utf-8') as file:
    json.dump(cleaned_data, file, ensure_ascii=False, indent=4)


import json

with open('kannada_sentences_cleaned.json', encoding='utf-8') as j:
    f = json.load(j)
    print(f)


    import json
import csv

# Read JSON data from file
with open('kannada_sentences_cleaned.json', 'r', encoding='utf-8') as jsonfile:
    data = json.load(jsonfile)

# Open a CSV file for writing
with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['word', 'sentence'])
    
    # Write the data
    for word, sentences in data.items():
        for sentence in sentences:
            csvwriter.writerow([word, sentence])

print("CSV file created successfully.")
import pandas as pd

df = pd.read_csv('output.csv')

new_df = df.drop_duplicates(subset=['sentence'], keep=False)

new_df.to_csv('kannada_polysemy_sentences.csv', index=False)
