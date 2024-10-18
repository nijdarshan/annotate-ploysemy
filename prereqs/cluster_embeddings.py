import pandas as pd
import numpy as np
import torch
from umap import UMAP
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from scipy.spatial.distance import cdist
import plotly.io as pio
import os
import gc
import glob

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_word_embeddings(sentences, words, tokenizer, model):
    print(f"Input sentences: {len(sentences)}")
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    word_embeddings = []
    valid_sentences = []
    valid_words = []

    for idx, (sentence, word) in enumerate(zip(sentences, words)):
        word_tokens = tokenizer.tokenize(word)
        word_ids = tokenizer.convert_tokens_to_ids(word_tokens)

        input_ids = inputs.input_ids[idx]
        word_positions = []

        for i in range(len(input_ids) - len(word_ids) + 1):
            if input_ids[i:i+len(word_ids)].tolist() == word_ids:
                word_positions.extend(range(i, i+len(word_ids)))

        if word_positions:
            word_embedding = outputs.last_hidden_state[idx, word_positions, :].mean(dim=0).cpu().numpy()
        else:
            word_embedding = outputs.last_hidden_state[idx].mean(dim=0).cpu().numpy()

        word_embeddings.append(word_embedding)
        valid_sentences.append(sentence)
        valid_words.append(word)

    print(f"Output valid sentences: {len(valid_sentences)}")
    return word_embeddings, valid_sentences, valid_words

def save_word_embeddings(word_model_dir, embeddings, sentences):
    os.makedirs(word_model_dir, exist_ok=True)
    
    for idx, (embedding, sentence) in enumerate(zip(embeddings, sentences)):
        embedding_file = os.path.join(word_model_dir, f"sentence_{idx+1}_embedding.csv")
        np.savetxt(embedding_file, embedding.reshape(1, -1), delimiter=",")
        
        sentence_file = os.path.join(word_model_dir, f"sentence_{idx+1}.txt")
        with open(sentence_file, "w", encoding="utf-8") as f:
            f.write(sentence)

def cluster_embeddings(embeddings, method='kmeans', n_clusters=2):
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters)
    else:
        raise ValueError("Invalid clustering method")
    return clusterer.fit_predict(embeddings), clusterer

def plot_clusters(result, model_name, word):
    fig = go.Figure()

    colors = ['#FF6B6B', '#4ECDC4']
    for i in range(2):
        mask = result['cluster_labels'] == i
        fig.add_trace(go.Scatter(
            x=result['reduced_embeddings'][mask, 0],
            y=result['reduced_embeddings'][mask, 1],
            mode='markers',
            marker=dict(color=colors[i], size=8),
            name=f'Cluster {i+1}',
            text=np.array(result['valid_sentences'])[mask],
            hoverinfo='text'
        ))

    fig.add_trace(go.Scatter(
        x=result['reduced_centers'][:, 0],
        y=result['reduced_centers'][:, 1],
        mode='markers',
        marker=dict(color='black', size=12, symbol='star'),
        name='Cluster Centroids'
    ))

    fig.update_layout(
        title=f'Word Embedding Clusters for "{word}" - {model_name}',
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Clusters",
        height=600,
        width=800
    )

    return fig
def process_word(word, unlabeled_df, model_name, cluster_method):
    unlabeled_word_df = unlabeled_df[unlabeled_df['word'] == word]
    unlabeled_sentences = unlabeled_word_df['sentence'].tolist()
    unlabeled_words = unlabeled_word_df['word'].tolist()

    print(f"Processing {word} with {model_name}: {len(unlabeled_sentences)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    word_embeddings, valid_sentences, valid_words = get_word_embeddings(unlabeled_sentences, unlabeled_words, tokenizer, model)

    print(f"After embedding: {len(valid_sentences)} valid sentences")

    all_embeddings = np.array(word_embeddings)

    all_predicted_labels, clusterer = cluster_embeddings(all_embeddings, method=cluster_method, n_clusters=2)

    reducer = UMAP(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    reduced_centers = reducer.transform(clusterer.cluster_centers_)

    result = {
        'embeddings': all_embeddings,
        'cluster_labels': all_predicted_labels,
        'valid_sentences': valid_sentences,
        'valid_words': valid_words,
        'cluster_centers': clusterer.cluster_centers_,
        'reduced_embeddings': reduced_embeddings,
        'reduced_centers': reduced_centers,
    }

    # Create directory for this word and model
    word_dir = os.path.join('results', word)
    model_dir = os.path.join(word_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save embeddings and other data
    np.save(os.path.join(model_dir, 'embeddings.npy'), all_embeddings)
    np.save(os.path.join(model_dir, 'cluster_labels.npy'), all_predicted_labels)
    np.save(os.path.join(model_dir, 'cluster_centers.npy'), clusterer.cluster_centers_)
    np.save(os.path.join(model_dir, 'reduced_embeddings.npy'), reduced_embeddings)
    
    # Save sentences with their corresponding indices
    with open(os.path.join(model_dir, 'sentences_with_indices.txt'), 'w', encoding='utf-8') as f:
        for idx, sentence in enumerate(valid_sentences):
            f.write(f"{idx}\t{sentence}\n")

    # Save a mapping of sentence indices to their embeddings
    sentence_embedding_map = {idx: embedding for idx, embedding in enumerate(all_embeddings)}
    np.save(os.path.join(model_dir, 'sentence_embedding_map.npy'), sentence_embedding_map)

    # Generate and save cluster plot
    cluster_plot = plot_clusters(result, model_name, word)
    pio.write_html(cluster_plot, file=os.path.join(model_dir, 'cluster_plot.html'))

    del model, tokenizer
    gc.collect()

    return result

def generate_word_report(word, results, output_file):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Word Analysis Report: {word}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .model-section {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Word Analysis Report: {word}</h1>
    """

    for model_name in results:
        html_content += f"""
        <div class="model-section">
            <h2>{model_name}</h2>
            <p>Number of embeddings: {len(results[model_name]['embeddings'])}</p>
            <p>Number of clusters: {len(set(results[model_name]['cluster_labels']))}</p>
            <p><a href="{model_name}/cluster_plot.html" target="_blank">View Cluster Plot</a></p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

def get_completed_words(results_dir):
    completed_words = []
    for word_dir in glob.glob(os.path.join(results_dir, '*')):
        if os.path.isdir(word_dir):
            word = os.path.basename(word_dir)
            if all_files_exist(word_dir):
                completed_words.append(word)
    return completed_words

def all_files_exist(word_dir):
    expected_files = ['word_analysis.html']
    for model_dir in glob.glob(os.path.join(word_dir, '*')):
        if os.path.isdir(model_dir):
            expected_files.extend([
                os.path.join(os.path.basename(model_dir), 'embeddings.npy'),
                os.path.join(os.path.basename(model_dir), 'cluster_labels.npy'),
                os.path.join(os.path.basename(model_dir), 'cluster_centers.npy'),
                os.path.join(os.path.basename(model_dir), 'reduced_embeddings.npy'),
                os.path.join(os.path.basename(model_dir), 'sentences_with_indices.txt'),
                os.path.join(os.path.basename(model_dir), 'sentence_embedding_map.npy'),
                os.path.join(os.path.basename(model_dir), 'cluster_plot.html')
            ])
    return all(os.path.exists(os.path.join(word_dir, file)) for file in expected_files)

def process_all_words(unlabeled_df, model_names, cluster_method):
    results_dir = 'test'
    os.makedirs(results_dir, exist_ok=True)
    
    completed_words = get_completed_words(results_dir)
    print(f"Already completed words: {completed_words}")
    
    unique_words = unlabeled_df['word'].unique()
    words_to_process = [word for word in unique_words if word not in completed_words]
    
    results = {}
    for word in words_to_process:
        word_results = {}
        print(f"\nProcessing word: {word}")
        
        word_dir = os.path.join(results_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        
        for model_name in model_names:
            model_dir = os.path.join(word_dir, model_name)
            if not os.path.exists(model_dir) or not all_files_exist(model_dir):
                result = process_word(word, unlabeled_df, model_name, cluster_method)
                word_results[model_name] = result
            else:
                print(f"Skipping {model_name} for {word} as it's already processed")
        
        if word_results:
            output_file = os.path.join(word_dir, f"{word}_analysis.html")
            generate_word_report(word, word_results, output_file)
        
        results[word] = word_results

    return results

if __name__ == "__main__":
    unlabeled_df = pd.read_csv('kannada_polysemy_sentences.csv')
    
    # please test on a smaller sample first
    #unlabeled_df_f =unlabeled_df[unlabeled_df.word == unlabeled_df.word.unique()[1]].sample(20)
    model_names = [
        'l3cube-pune/kannada-bert',
        'google/muril-base-cased',
        'pierluigic/xl-lexeme'
    ]
    cluster_method = 'kmeans'

    results = process_all_words(unlabeled_df, model_names, cluster_method)

    print("Word analysis complete. Results are available in individual word directories.")