# Prerequisites for Polysemy Annotation Application

This document provides a detailed explanation of the prerequisites and processes involved in setting up and running the `annotate-polysemy` application. The application involves several steps, including fetching sentences, cleaning them, and clustering embeddings. Below is a step-by-step guide to understanding and executing these processes.

## 1. Fetching Random Sentences

The first step involves using the `fetch_random_sentences.py` script to extract sentences containing specific words from a large corpus. This script is essential for preparing the data that will be used in subsequent steps.

### Key Steps:
- **Input**: A text corpus file (`corpus_path`) and a list of target words (`word_list`).
- **Output**: A JSON file (`output_file`) containing sentences for each word.
- **Process**:
  - The script reads chunks of the corpus file and searches for sentences containing the target words.
  - It uses regular expressions to identify word occurrences and stores the sentences in a dictionary.
  - The script continues until the desired number of sentences (`num_sentences`) is found for each word or the maximum number of attempts is reached.

### Code Reference:
```python:prereqs/fetch_random_sentences.py
startLine: 7
endLine: 94
```

## 2. Cleaning Sentences

Once the sentences are fetched, the `clean_sentences.py` script is used to preprocess and clean these sentences. This step ensures that the data is in a suitable format for embedding and clustering.

### Key Steps:
- **Input**: The JSON file generated from the previous step.
- **Output**: A cleaned JSON file and a CSV file with unique sentences.
- **Process**:
  - The script removes unwanted characters, symbols, and patterns from the sentences.
  - It filters out sentences that do not meet specific criteria, such as those containing underscores or starting with certain words.
  - The cleaned sentences are then saved in a new JSON file and converted to a CSV format for further processing.

### Code Reference:
```python:prereqs/clean_sentences.py
startLine: 6
endLine: 109
```

## 3. Clustering Embeddings

The final step involves the `cluster_embeddings.py` script, which processes the cleaned sentences to generate and cluster word embeddings. This step is crucial for visualizing and analyzing the semantic relationships between words.

### Key Steps:
- **Input**: The CSV file with cleaned sentences.
- **Output**: Clustered embeddings and visualizations.
- **Process**:
  - The script uses pre-trained language models to generate embeddings for each sentence.
  - It applies clustering algorithms (e.g., KMeans) to group similar embeddings.
  - The results are visualized using UMAP and Plotly, and the data is saved for further analysis.

### Code Reference:
```python:prereqs/cluster_embeddings.py
startLine: 20
endLine: 274
```

## Conclusion

By following these steps and ensuring all prerequisites are met, you will be able to successfully set up and run the `annotate-polysemy` application. Each script plays a vital role in preparing and processing the data, ultimately enabling the application to perform semantic analysis and visualization.
