# annotate-ploysemy

## Overview
`annotate-ploysemy` is a Streamlit application designed to visualize sentence embeddings and facilitate the manual annotation of sentences into clusters. The application uses scatter plots to display embeddings and allows users to interactively select and categorize sentences.

## Features
- **Upload Embeddings**: Upload `.npy` files containing reduced embeddings and cluster labels.
- **Upload Sentences**: Upload a `.csv` file containing sentences with their indices.
- **Interactive Visualization**: Visualize sentence embeddings in a scatter plot and interactively select sentences.
- **Manual Annotation**: Add selected sentences to Cluster 1 or Cluster 2.
- **Save Annotations**: Save the annotated sentences to a CSV file.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/annotate-ploysemy.git
    cd annotate-ploysemy
    ```
2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    venv\Scripts\activate  # On Windows
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```
2. Upload the required files:
    - `reduced_embeddings.npy`: Numpy file containing the reduced embeddings.
    - `sentences_with_indices.csv`: CSV file containing sentences with their indices.
    - `cluster_labels.npy`: Numpy file containing cluster labels.
3. Interact with the scatter plot to select and annotate sentences.
4. Save the annotations to a CSV file.

## File Descriptions
- `app.py`: Main application file.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Example Files
- `reduced_embeddings.npy`: Example embeddings file.
- `sentences_with_indices.csv`: Example sentences file.
- `cluster_labels.npy`: Example cluster labels file.

## License
This project is licensed under the MIT License.# annotate-ploysemy