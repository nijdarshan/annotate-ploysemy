import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
from database import Database
import auth
import os
from google.cloud import storage

# Color-blind friendly palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Database configuration
config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": 5432
}
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
                        "opacity": 0.7  # Set fixed opacity for consistency
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
        "xAxis": {"name": "Dimension 1"},
        "yAxis": {"name": "Dimension 2"},
        "series": series_data,
        "dataZoom": [{"type": "slider"}, {"type": "inside"}]
    }

    return options

# Show the user login screen
def show_login_screen():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if auth.login(db, username, password):
            st.session_state.user = username
            st.session_state.page = 'word_list'
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
    
    # Add a button to navigate to the registration screen
    if st.button("Register"):
        st.session_state.page = 'register'
        st.experimental_rerun()

# Show the registration screen
def show_registration_screen():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if auth.register(db, username, password):
            st.success("Registration successful! Please log in.")
            st.session_state.page = 'login'
            st.experimental_rerun()
        else:
            st.error("Registration failed. Please try again.")

# Show the word list screen
def show_word_list():
    st.title("Polysemy Annotation - Word List")
    
    # Toggle word list view
    if st.button("Toggle Word List View"):
        st.session_state.word_list_toggle = db.toggle_word_list(st.session_state.user_id)
        st.experimental_rerun()

    # Fetch word directories from the 'data/' directory
    word_dirs = os.listdir('data')

    for word_dir in word_dirs:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"📝 {word_dir}"):
                st.session_state.current_word = word_dir
                st.session_state.page = 'annotation'
                st.experimental_rerun()
        with col2:
            st.write("Completed" if is_word_completed(word_dir) else "Not completed")

def is_word_completed(word_dir):
    embeddings, sentences, cluster_labels = load_word_data(word_dir)
    total_sentences = len(sentences)
    clustered_sentences = len(st.session_state.cluster1) + len(st.session_state.cluster2)
    return clustered_sentences == total_sentences

# Show the annotation screen
def show_annotation_screen():
    if 'current_word' not in st.session_state:
        st.session_state.page = 'word_list'
        st.experimental_rerun()

    word_dir = st.session_state.current_word
    embeddings, sentences, cluster_labels = load_word_data(word_dir)
    options = create_chart_options(embeddings, sentences, cluster_labels, st.session_state.used_indices)

    col1, col2 = st.columns([4, 1])
    with col1:
        events = {"click": "function(params) { return [params.name, params.dataIndex]; }"}
        selected_point = st_echarts(options=options, events=events, height=600, width="100%")

    with col2:
        if selected_point:
            st.session_state.selected_sentence, st.session_state.selected_index = selected_point
            st.write(st.session_state.selected_sentence)

        if st.session_state.selected_sentence:
            if st.button("Add to Cluster 1"):
                if st.session_state.selected_index not in st.session_state.used_indices:
                    st.session_state.cluster1.append((st.session_state.selected_index, st.session_state.selected_sentence))
                    st.session_state.used_indices.add(st.session_state.selected_index)
                    st.experimental_rerun()

            if st.button("Add to Cluster 2"):
                if st.session_state.selected_index not in st.session_state.used_indices:
                    st.session_state.cluster2.append((st.session_state.selected_index, st.session_state.selected_sentence))
                    st.session_state.used_indices.add(st.session_state.selected_index)
                    st.experimental_rerun()

    st.subheader("Current Clusters")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Cluster 1:")
        for i, (index, sentence) in enumerate(st.session_state.cluster1):
            st.write(f"{i + 1}. {sentence}")
            if st.button(f"Delete from Cluster 1", key=f"del1_{i}"):
                st.session_state.cluster1.pop(i)
                st.session_state.used_indices.remove(index)
                st.experimental_rerun()

    with col2:
        st.write("Cluster 2:")
        for i, (index, sentence) in enumerate(st.session_state.cluster2):
            st.write(f"{i + 1}. {sentence}")
            if st.button(f"Delete from Cluster 2", key=f"del2_{i}"):
                st.session_state.cluster2.pop(i)
                st.session_state.used_indices.remove(index)
                st.experimental_rerun()

    if st.button("Save Annotations"):
        df = pd.DataFrame({
            'Index': [item[0] for item in st.session_state.cluster1 + st.session_state.cluster2],
            'Sentence': [item[1] for item in st.session_state.cluster1 + st.session_state.cluster2],
            'Cluster': ['Cluster 1']*len(st.session_state.cluster1) + ['Cluster 2']*len(st.session_state.cluster2)
        })
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name=f"{st.session_state.current_word}_annotations.csv")

    if st.button("← Back to Word List"):
        st.session_state.page = 'word_list'
        st.experimental_rerun()

# Load word data from the 'data/' directory
def load_word_data(word_name):
    base_path = f'data/{word_name}'
    if os.getenv("USE_GCS", "false").lower() == "true":
        # Load from Google Cloud Storage
        bucket_name = '<BUCKET-NAME>'
        download_blob(bucket_name, f'{base_path}/reduced_embeddings.csv', '/tmp/reduced_embeddings.csv')
        download_blob(bucket_name, f'{base_path}/sentences_with_indices.csv', '/tmp/sentences_with_indices.csv')
        download_blob(bucket_name, f'{base_path}/cluster_labels.csv', '/tmp/cluster_labels.csv')
    else:
        # Load from local storage
        local_path = f'./{base_path}'
        os.makedirs('/tmp', exist_ok=True)
        os.system(f'cp {local_path}/reduced_embeddings.csv /tmp/reduced_embeddings.csv')
        os.system(f'cp {local_path}/sentences_with_indices.csv /tmp/sentences_with_indices.csv')
        os.system(f'cp {local_path}/cluster_labels.csv /tmp/cluster_labels.csv')

    embeddings = pd.read_csv('/tmp/reduced_embeddings.csv')[['Dim1', 'Dim2']].values
    sentences = pd.read_csv('/tmp/sentences_with_indices.csv').set_index('Index')['Sentence'].to_dict()
    cluster_labels = pd.read_csv('/tmp/cluster_labels.csv')['Cluster'].values

    return embeddings, sentences, cluster_labels

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

# Main function to manage navigation between pages
def main():
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
    if 'word_list_toggle' not in st.session_state:
        st.session_state.word_list_toggle = False

    if st.session_state.page == 'login':
        show_login_screen()
    elif st.session_state.page == 'register':
        show_registration_screen()
    elif st.session_state.page == 'word_list':
        show_word_list()
    elif st.session_state.page == 'annotation':
        show_annotation_screen()

if __name__ == "__main__":
    main()
