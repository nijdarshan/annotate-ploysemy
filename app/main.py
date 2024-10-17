import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
from database import Database
import auth
import os
import random
from google.cloud import storage

# Color-blind friendly palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Database configuration
config = {
    "dbname": "polysemy_db",
    "user": "nij",
    "password": "nijD22",
    "host": "localhost",
    "port": 5432  # Default PostgreSQL port
}

# Initialize the Database
db = Database(config)

# Function to create chart options for visualization
def create_chart_options(embeddings, sentences, cluster_labels, used_indices):
    series_data = []
    unique_clusters = pd.unique(cluster_labels)
    
    for cluster in unique_clusters:
        cluster_indices = cluster_labels == cluster
        cluster_data = []
        
        for idx, is_in_cluster in enumerate(cluster_indices):
            if is_in_cluster and idx not in used_indices:
                point = embeddings[idx]
                sentence = sentences.get(idx, f"Sentence {idx}")
                cluster_data.append({
                    "value": point.tolist(),
                    "name": sentence,
                    "itemStyle": {
                        "color": COLORS[int(cluster) % len(COLORS)],
                        "opacity": random.uniform(0.5, 1)  # Random opacity for differentiation
                    }
                })

        series_data.append({
            "name": f"Cluster {cluster}",
            "type": "scatter",
            "data": cluster_data,
            "symbolSize": 10
        })

    options = {
        "title": {"text": "Sentence Embeddings Visualization"},
        "tooltip": {
            "trigger": "item",
            "formatter": "<div style='white-space: normal; max-width: 300px;'>{b}</div>",
            "renderMode": "html",
        },
        "xAxis": {"name": "Dimension 1"},  # X-axis label
        "yAxis": {"name": "Dimension 2"},  # Y-axis label
        "series": series_data,
        "dataZoom": [
            {
                "type": "slider",
                "show": True,
                "start": 0,
                "end": 100
            },
            {
                "type": "inside",
                "start": 0,
                "end": 100
            }
        ]
    }

    return options

# Show the user login screen
def show_login_screen():
    st.title("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        if auth.login(db, username, password):
            st.session_state.user = username
            st.session_state.page = 'word_list'
            st.success("Login successful!")
            st.rerun()  # Ensure immediate rerun
        else:
            st.error("Invalid username or password")

    st.write("Don't have an account?")
    if st.button("Create New Account", key="create_account_button"):
        st.session_state.page = 'register'
        st.rerun()

# Show the registration screen
def show_register_screen():
    st.title("Create New Account")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    
    if st.button("Register"):
        auth.register(db, username, password)
        st.session_state.page = 'login'

    if st.button("Back to Login"):
        st.session_state.page = 'login'

# Show the word list screen (Read from data/ directory)
def show_word_list():
    st.title("Polysemy Annotation - Word List")
    
    # Fetch word directories from the 'data/' directory
    word_dirs = os.listdir('data')

    for word_dir in word_dirs:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"📝 {word_dir}", key=f"word_{word_dir}"):
                st.session_state.current_word = word_dir
                st.session_state.page = 'annotation'
                st.rerun()  # Use st.rerun() for immediate rerun
        with col2:
            # Check if the word is completed
            if is_word_completed(word_dir):
                st.write("Completed")
            else:
                st.write("Not completed")

def is_word_completed(word_dir):
    # Load the data for the word
    embeddings, sentences, cluster_labels = load_word_data(word_dir)
    
    # Check if all sentences have been clustered
    total_sentences = len(sentences)
    clustered_sentences = len(st.session_state.cluster1) + len(st.session_state.cluster2)
    
    return clustered_sentences == total_sentences

# Show the annotation screen for the selected word
def show_annotation_screen():
    if 'current_word' not in st.session_state:
        st.session_state.page = 'word_list'
        st.rerun()
        return

    word_dir = st.session_state.current_word
    embeddings, sentences, cluster_labels = load_word_data(word_dir)

    options = create_chart_options(embeddings, sentences, cluster_labels, st.session_state.used_indices)

    # Layout adjustment: Move plot to the left and controls to the right
    col1, col2 = st.columns([4, 1])
    with col1:
        # Create a container for the plot
        plot_container = st.container()
        with plot_container:
            # Handle visualization and annotation selection
            events = {"click": "function(params) { return [params.name, params.dataIndex]; }"}
            selected_point = st_echarts(options=options, events=events, height=600, width="100%", key="echarts")

    with col2:
        if selected_point:
            st.session_state.selected_sentence, st.session_state.selected_index = selected_point
            st.subheader("Selected sentence:")
            st.write(st.session_state.selected_sentence)

        if st.session_state.selected_sentence:
            if st.button("Add to Cluster 1", key="add_cluster1"):
                if st.session_state.selected_index not in st.session_state.used_indices:
                    st.session_state.cluster1.append((st.session_state.selected_index, st.session_state.selected_sentence))
                    st.session_state.used_indices.add(st.session_state.selected_index)
                    st.rerun()

            if st.button("Add to Cluster 2", key="add_cluster2"):
                if st.session_state.selected_index not in st.session_state.used_indices:
                    st.session_state.cluster2.append((st.session_state.selected_index, st.session_state.selected_sentence))
                    st.session_state.used_indices.add(st.session_state.selected_index)
                    st.rerun()

    # Display current clusters with delete buttons
    st.subheader("Current Clusters")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Cluster 1:")
        for i, (index, sentence) in enumerate(st.session_state.cluster1):
            st.write(f"{i + 1}. {sentence}")
            if st.button(f"Delete from Cluster 1", key=f"del1_{i}"):
                st.session_state.cluster1.pop(i)
                st.session_state.used_indices.remove(index)
                st.rerun()

    with col2:
        st.write("Cluster 2:")
        for i, (index, sentence) in enumerate(st.session_state.cluster2):
            st.write(f"{i + 1}. {sentence}")
            if st.button(f"Delete from Cluster 2", key=f"del2_{i}"):
                st.session_state.cluster2.pop(i)
                st.session_state.used_indices.remove(index)
                st.rerun()

    # Save and download annotations
    if st.button("Save Annotations"):
        df = pd.DataFrame({
            'Index': [item[0] for item in st.session_state.cluster1 + st.session_state.cluster2],
            'Sentence': [item[1] for item in st.session_state.cluster1 + st.session_state.cluster2],
            'Cluster': ['Cluster 1']*len(st.session_state.cluster1) + ['Cluster 2']*len(st.session_state.cluster2)
        })
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{st.session_state.current_word}_annotated_sentences.csv",
            mime="text/csv"
        )

    if st.button("← Back to Word List", key="back_to_word_list"):
        st.session_state.page = 'word_list'
        st.rerun()

# Load word data from the 'data/' directory
def load_word_data(word_name):
    base_path = f'data/{word_name}'

    # Download files from the bucket
    bucket_name = 'anp-bkt'
    download_blob(bucket_name, f'{base_path}/reduced_embeddings.csv', '/tmp/reduced_embeddings.csv')
    download_blob(bucket_name, f'{base_path}/sentences_with_indices.csv', '/tmp/sentences_with_indices.csv')
    download_blob(bucket_name, f'{base_path}/cluster_labels.csv', '/tmp/cluster_labels.csv')

    # Load embeddings
    embeddings_df = pd.read_csv('/tmp/reduced_embeddings.csv')
    embeddings = embeddings_df[['Dim1', 'Dim2']].values

    # Load sentences
    sentences_df = pd.read_csv('/tmp/sentences_with_indices.csv')
    sentences = {index: row['Sentence'] for index, row in sentences_df.iterrows()}

    # Load cluster labels
    cluster_labels_df = pd.read_csv('/tmp/cluster_labels.csv')
    cluster_labels = cluster_labels_df['Cluster'].values

    return embeddings, sentences, cluster_labels

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# Main function to manage navigation between pages
def main():
    # Initialize session state variables
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'used_indices' not in st.session_state:
        st.session_state.used_indices = set()
    if 'cluster1' not in st.session_state:
        st.session_state.cluster1 = []
    if 'cluster2' not in st.session_state:
        st.session_state.cluster2 = []
    if 'selected_sentence' not in st.session_state:
        st.session_state.selected_sentence = None
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = None

    # Simple navigation flow
    if st.session_state.page == 'login':
        show_login_screen()
    elif st.session_state.page == 'register':
        show_register_screen()
    elif st.session_state.page == 'word_list':
        show_word_list()
    elif st.session_state.page == 'annotation':
        show_annotation_screen()

if __name__ == "__main__":
    main()
