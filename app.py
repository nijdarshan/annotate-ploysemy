import streamlit as st
import numpy as np
import pandas as pd
from streamlit_echarts import st_echarts

# Color-blind friendly palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def load_embeddings(file):
    return np.load(file)

def load_sentences(file):
    df = pd.read_csv(file, sep=',', header=0)
    sentences = {row['Index']: row['Sentence'] for _, row in df.iterrows()}
    return sentences

def load_cluster_labels(file):
    return np.load(file)

def create_chart_options(embeddings, sentences, cluster_labels, used_indices):
    series_data = []
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_data = []
        for idx in cluster_indices:
            if idx not in used_indices:
                point = embeddings[idx]
                sentence = sentences[idx]
                cluster_data.append({
                    "value": point.tolist(),
                    "name": sentence
                })

        series_data.append({
            "name": f"Cluster {cluster}",
            "type": "scatter",
            "data": cluster_data,
            "symbolSize": 10,
            "itemStyle": {
                "color": COLORS[int(cluster) % len(COLORS)]
            }
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

def main():
    st.set_page_config(layout="wide")
    st.title('Sentence Embeddings Visualizer')

    # File uploaders
    embeddings_file = st.file_uploader("Upload reduced_embeddings.npy", type="npy")
    sentences_file = st.file_uploader("Upload sentences_with_indices.csv", type="csv")
    cluster_labels_file = st.file_uploader("Upload cluster_labels.npy", type="npy")

    # Initialize session state variables
    if 'cluster1' not in st.session_state:
        st.session_state.cluster1 = []
    if 'cluster2' not in st.session_state:
        st.session_state.cluster2 = []
    if 'used_indices' not in st.session_state:
        st.session_state.used_indices = set()
    if 'selected_sentence' not in st.session_state:
        st.session_state.selected_sentence = None

    # Check if all files are uploaded
    if embeddings_file and sentences_file and cluster_labels_file:
        embeddings = load_embeddings(embeddings_file)
        sentences = load_sentences(sentences_file)
        cluster_labels = load_cluster_labels(cluster_labels_file)

        if len(embeddings) > 0 and len(sentences) > 0 and len(cluster_labels) > 0:
            options = create_chart_options(embeddings, sentences, cluster_labels, st.session_state.used_indices)

            # Adjust event handling to return a simple, serializable value
            events = {
                "click": "function(params) { return params.name; }"
            }

            selected_point = st_echarts(options=options, events=events, height="600px")

            if selected_point:
                st.session_state.selected_sentence = selected_point

            if st.session_state.selected_sentence:
                st.subheader("Selected sentence:")
                st.write(st.session_state.selected_sentence)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Add to Cluster 1"):
                        index = next((i for i, s in sentences.items() if s == st.session_state.selected_sentence), None)
                        if index is not None and index not in st.session_state.used_indices:
                            st.session_state.cluster1.append((index, st.session_state.selected_sentence))
                            st.session_state.used_indices.add(index)
                            st.rerun()  # Use st.rerun() to refresh the app state
                with col2:
                    if st.button("Add to Cluster 2"):
                        index = next((i for i, s in sentences.items() if s == st.session_state.selected_sentence), None)
                        if index is not None and index not in st.session_state.used_indices:
                            st.session_state.cluster2.append((index, st.session_state.selected_sentence))
                            st.session_state.used_indices.add(index)
                            st.rerun()  # Use st.rerun() to refresh the app state

            # Display the contents of cluster1 and cluster2
            st.write("Cluster 1 Contents:", st.session_state.cluster1)
            st.write("Cluster 2 Contents:", st.session_state.cluster2)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cluster 1 Sentences")
                for i, item in enumerate(st.session_state.cluster1):
                    if isinstance(item, tuple) and len(item) == 2:
                        index, sentence = item
                        st.write(f"{i + 1}. {sentence}")
                        if st.button(f"Delete from Cluster 1", key=f"del1_{i}"):
                            st.session_state.cluster1.pop(i)
                            st.session_state.used_indices.remove(index)
                            st.rerun()  # Use st.rerun() to refresh the app state
                    else:
                        st.error("Unexpected item in Cluster 1")

            with col2:
                st.subheader("Cluster 2 Sentences")
                for i, item in enumerate(st.session_state.cluster2):
                    if isinstance(item, tuple) and len(item) == 2:
                        index, sentence = item
                        st.write(f"{i + 1}. {sentence}")
                        if st.button(f"Delete from Cluster 2", key=f"del2_{i}"):
                            st.session_state.cluster2.pop(i)
                            st.session_state.used_indices.remove(index)
                            st.rerun()  # Use st.rerun() to refresh the app state
                    else:
                        st.error("Unexpected item in Cluster 2")

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
                    file_name="annotated_sentences.csv",
                    mime="text/csv"
                )
        else:
            st.error("Error: Unable to load data from the uploaded files.")
    else:
        st.info("Please upload all three files to visualize the embeddings.")

if __name__ == "__main__":
    main()
