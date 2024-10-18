import json
import random
import re
from collections import defaultdict
import os

def fetch_sentences(corpus_path, word_list, num_sentences, output_file, chunk_size=10000000, max_attempts=1000000):
    word_sentences = defaultdict(list)
    word_patterns = {word: re.compile(r'(?<!\S)' + re.escape(word) + r'(?!\S)', re.UNICODE | re.IGNORECASE) for word in word_list}
    
    if not os.path.exists(corpus_path):
        print(f"Error: File '{corpus_path}' does not exist.")
        return

    file_size = os.path.getsize(corpus_path)
    print(f"File size: {file_size} bytes")

    attempts = 0
    lines_processed = 0

    try:
        with open(corpus_path, 'rb') as file:
            print("File opened successfully.")
            
            while attempts < max_attempts:
                sentences_found = [len(word_sentences[word]) for word in word_list]
                print(f"Current sentences found: {dict(zip(word_list, sentences_found))}")
                
                if all(count >= num_sentences for count in sentences_found):
                    print("All required sentences found. Exiting loop.")
                    break

                start_pos = random.randint(0, max(0, file_size - chunk_size))
                print(f"Attempt {attempts + 1}: Starting at position {start_pos}")
                file.seek(start_pos)
                
                chunk = file.read(chunk_size)
                try:
                    chunk = chunk.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    print(f"Error decoding chunk at position {start_pos}. Skipping.")
                    attempts += 1
                    continue

                # Split into sentences
                sentences = re.split(r'(?<=[.।॥?!])\s+|\*\\[*n]|\n+', chunk)
                print(f"Read {len(sentences)} sentences in this chunk.")
                
                for sentence in sentences:
                    lines_processed += 1
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    for word, pattern in word_patterns.items():
                        if len(word_sentences[word]) >= num_sentences:
                            continue
                        
                        if pattern.search(sentence):
                            word_sentences[word].append(sentence)
                            print(f"Found match for '{word}': {sentence[:50]}...")  # Print first 50 chars
                            if len(word_sentences[word]) >= num_sentences:
                                break

                attempts += 1

                if attempts % 10 == 0 or attempts == 1:
                    print(f"Processed approximately {lines_processed} sentences in {attempts} attempts.")
                    for word, sentences in word_sentences.items():
                        print(f"  '{word}': {len(sentences)} sentences")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Trim excess sentences and shuffle
    for word in word_sentences:
        if len(word_sentences[word]) > num_sentences:
            word_sentences[word] = random.sample(word_sentences[word], num_sentences)
        random.shuffle(word_sentences[word])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(word_sentences, f, ensure_ascii=False, indent=2)

    print(f"Processed approximately {lines_processed} sentences in {attempts} attempts.")
    for word, sentences in word_sentences.items():
        print(f"Found {len(sentences)} sentences for '{word}'")
        

# Example usage
corpus_path = 'kn_1.txt'
word_list = [] #list of your words - could add feature to read from .txt or .csv
num_sentences = 750
output_file = 'kannada_sentences_new2.json'
fetch_sentences(corpus_path, word_list, num_sentences, output_file)
