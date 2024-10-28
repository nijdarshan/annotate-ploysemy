import pandas as pd
import numpy as np
import torch
from umap import UMAP
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import plotly.io as pio
import os
import gc
import glob

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_color_palette(n_colors):
    """Generate a color palette for visualization"""
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    # If we need more colors than available, cycle through them
    return [colors[i % len(colors)] for i in range(n_colors)]

def remap_cluster_labels(true_labels, predicted_labels):
    """
    Remap cluster labels to best match true labels using Hungarian algorithm
    """
    # Remove samples with label 0 or NaN
    valid_mask = (true_labels != 0) & ~np.isnan(true_labels)
    true_labels_valid = true_labels[valid_mask]
    predicted_labels_valid = predicted_labels[valid_mask]
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels_valid, predicted_labels_valid)
    
    # Use Hungarian algorithm to find optimal mapping
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Create mapping dictionary
    label_mapping = {old: new for old, new in zip(col_ind, row_ind)}
    
    # Remap predicted labels
    remapped_labels = np.array([label_mapping.get(label, label) for label in predicted_labels])
    
    return remapped_labels

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate accuracy and precision metrics after removing ignored labels (0) and NaN
    """
    # Remove samples with label 0 or NaN
    valid_mask = (true_labels != 0) & ~np.isnan(true_labels)
    true_labels_valid = true_labels[valid_mask]
    predicted_labels_valid = predicted_labels[valid_mask]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels_valid, predicted_labels_valid)
    precision = precision_score(true_labels_valid, predicted_labels_valid, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'confusion_matrix': confusion_matrix(true_labels_valid, predicted_labels_valid)
    }

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate accuracy and precision metrics after removing ignored labels (0)
    """
    # Remove samples with label 0
    valid_mask = true_labels != 0
    true_labels_valid = true_labels[valid_mask]
    predicted_labels_valid = predicted_labels[valid_mask]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels_valid, predicted_labels_valid)
    precision = precision_score(true_labels_valid, predicted_labels_valid, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'confusion_matrix': confusion_matrix(true_labels_valid, predicted_labels_valid)
    }

def get_word_embeddings(sentences, words, tokenizer, model):
    print(f"Input sentences: {len(sentences)}")
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    word_embeddings = []
    valid_sentences = []
    valid_words = []

    for idx, (sentence, word) in enumerate(zip(sentences, words)):
        word_tokens = tokenizer.tokenize(word)
        word_ids = tokenizer.convert_tokens_to_ids(word_tokens)

        input_ids = inputs['input_ids'][idx]
        word_positions = []

        current_input_ids = input_ids.cpu().tolist()
        
        for i in range(len(current_input_ids) - len(word_ids) + 1):
            if current_input_ids[i:i+len(word_ids)] == word_ids:
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

def cluster_embeddings(embeddings, true_labels):
    """Cluster embeddings based on number of unique non-zero labels"""
    unique_labels = np.unique(true_labels[true_labels != 0])
    n_clusters = len(unique_labels)
    print(f"Clustering into {n_clusters} clusters")
    
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = clusterer.fit_predict(embeddings)
    
    # Remap cluster labels to match true labels
    remapped_labels = remap_cluster_labels(true_labels, predicted_labels)
    
    return remapped_labels, clusterer

def plot_clusters(result, model_name, word):
    fig = go.Figure()

    # Get unique non-zero labels
    unique_labels = np.unique(result['true_labels'][result['true_labels'] != 0])
    colors = get_color_palette(len(unique_labels))
    
    # Plot points
    for i, label in enumerate(unique_labels):
        # Plot true labels
        mask = result['true_labels'] == label
        fig.add_trace(go.Scatter(
            x=result['reduced_embeddings'][mask, 0],
            y=result['reduced_embeddings'][mask, 1],
            mode='markers',
            marker=dict(color=colors[i], size=8, symbol='circle'),
            name=f'True Class {label}',
            text=np.array(result['valid_sentences'])[mask],
            hoverinfo='text'
        ))
        
        # Plot predicted clusters
        mask_pred = result['cluster_labels'] == label
        fig.add_trace(go.Scatter(
            x=result['reduced_embeddings'][mask_pred, 0],
            y=result['reduced_embeddings'][mask_pred, 1],
            mode='markers',
            marker=dict(color=colors[i], size=12, symbol='x'),
            name=f'Predicted Class {label}',
            text=np.array(result['valid_sentences'])[mask_pred],
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f'Word Embedding Clusters for "{word}" - {model_name}<br>Accuracy: {result["metrics"]["accuracy"]:.3f}, Precision: {result["metrics"]["precision"]:.3f}',
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Classes",
        height=800,
        width=1000
    )

    return fig

def process_word(word, labeled_df, model_name):
    word_df = labeled_df[labeled_df['word'] == word]
    sentences = word_df['sentence'].tolist()
    words = word_df['word'].tolist()
    true_labels = word_df['sense'].to_numpy()

    print(f"Processing {word} with {model_name}: {len(sentences)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    word_embeddings, valid_sentences, valid_words = get_word_embeddings(sentences, words, tokenizer, model)

    print(f"After embedding: {len(valid_sentences)} valid sentences")

    all_embeddings = np.array(word_embeddings)
    
    # Cluster and remap labels
    predicted_labels, clusterer = cluster_embeddings(all_embeddings, true_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predicted_labels)
    print(f"Metrics for {word} with {model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    
    # Reduce dimensionality for visualization
    reducer = UMAP(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(all_embeddings)

    result = {
        'embeddings': all_embeddings,
        'cluster_labels': predicted_labels,
        'true_labels': true_labels,
        'valid_sentences': valid_sentences,
        'valid_words': valid_words,
        'reduced_embeddings': reduced_embeddings,
        'metrics': metrics
    }

    # Create directory for this word and model
    word_dir = os.path.join('results', word)
    model_dir = os.path.join(word_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save results
    np.save(os.path.join(model_dir, 'embeddings.npy'), all_embeddings)
    np.save(os.path.join(model_dir, 'predicted_labels.npy'), predicted_labels)
    np.save(os.path.join(model_dir, 'true_labels.npy'), true_labels)
    np.save(os.path.join(model_dir, 'reduced_embeddings.npy'), reduced_embeddings)
    
    # Save sentences with their corresponding indices and labels
    with open(os.path.join(model_dir, 'sentences_with_labels.txt'), 'w', encoding='utf-8') as f:
        for idx, (sentence, true_label, pred_label) in enumerate(zip(valid_sentences, true_labels, predicted_labels)):
            f.write(f"{idx}\t{sentence}\t{true_label}\t{pred_label}\n")

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
            .metrics {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Word Analysis Report: {word}</h1>
    """

    for model_name, result in results.items():
        metrics = result['metrics']
        html_content += f"""
        <div class="model-section">
            <h2>{model_name}</h2>
            <div class="metrics">
                <p><strong>Accuracy:</strong> {metrics['accuracy']:.3f}</p>
                <p><strong>Precision:</strong> {metrics['precision']:.3f}</p>
            </div>
            <p><a href="{model_name}/cluster_plot.html" target="_blank">View Cluster Plot</a></p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

def process_all_words(labeled_df, model_names):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    unique_words = labeled_df['word'].unique()
    results = {}
    
    for word in unique_words:
        word_results = {}
        print(f"\nProcessing word: {word}")
        
        word_dir = os.path.join(results_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        
        for model_name in model_names:
            result = process_word(word, labeled_df, model_name)
            word_results[model_name] = result
        
        output_file = os.path.join(word_dir, f"word_analysis.html")
        generate_word_report(word, word_results, output_file)
        
        results[word] = word_results

    return results

if __name__ == "__main__":
    labeled_df = pd.read_csv('onhnd.csv')
    
    # For testing, use a smaller sample
    #labeled_df_f = labeled_df[labeled_df.word == labeled_df.word.unique()[0]].sample(30)
    
    model_names = [
        'l3cube-pune/kannada-bert',
        'google/muril-base-cased',
        'pierluigic/xl-lexeme'
    ]

    results = process_all_words(labeled_df, model_names)
    print("Word analysis complete. Results are available in individual word directories.")
