# Polysemy Annotation Application

## Overview
`annotate-ploysemy` is a Streamlit application designed to visualize sentence embeddings and facilitate the manual annotation of sentences into clusters. The application uses scatter plots to display embeddings and allows users to interactively select and categorize sentences.

## Prerequisites
To fetch, clean, and cluster polysemy word embeddings, check out [how](https://github.com/nijdarshan/annotate-ploysemy/tree/main/prereqs)

## Features

- **User Registration and Login**: Users can register and log in to access their personalized word list and annotation progress.
- **Word List Toggle**: Users can toggle the visibility of their word list.
- **Annotation Interface**: Users can view sentence embeddings, cluster sentences, and save their annotations.
- **Data Persistence**: User preferences and annotations are stored in a PostgreSQL database.
- **Flexible Data Source**: Supports loading data from local storage or Google Cloud Storage.

## Prerequisites

- **Python 3.9 or later**: Ensure Python is installed on your system.
- **Docker**: Required for containerizing the application.
- **Google Cloud SDK**: For deploying to Google Cloud Run.
- **PostgreSQL**: A running instance for the database.
- **Data Files**: Generate necessary data files, including sentence embeddings and cluster labels. These files should be stored either locally or in a Google Cloud Storage bucket.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up the Database**:
   - Ensure you have a PostgreSQL database running.
   - Execute the SQL script `schema.sql` to set up the necessary tables:
     ```bash
     psql -U <username> -d <database> -f schema.sql
     ```

3. **Environment Variables**:
   - Set the following environment variables for database configuration:
     - `DB_NAME`
     - `DB_USER`
     - `DB_PASSWORD`
     - `DB_HOST`
   - To use Google Cloud Storage, set `USE_GCS=true` and configure Google Cloud credentials.

4. **Install Dependencies**:
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

5. **Build the Docker Image**:
   - Ensure Docker is installed on your system.
   - Build the Docker image:
     ```bash
     docker build -t polysemy-annotation-app .
     ```

6. **Run the Application Locally**:
   - Start the application using Docker:
     ```bash
     docker run -p 8080:8080 polysemy-annotation-app
     ```

## Running Locally vs. Google Cloud

- **Local Storage**: By default, the application loads data from local storage. Ensure your data files are in the `data/` directory.
- **Google Cloud Storage**: To load data from GCS, set the `USE_GCS` environment variable to `true` and ensure your Google Cloud credentials are configured.

## Deployment on Google Cloud

To deploy this application on Google Cloud, follow these steps:

1. **Set Up Google Cloud Project**:
   - Create a new project in Google Cloud Console.
   - Enable the necessary APIs, such as Google Cloud Storage and Cloud Run.

2. **Configure Google Cloud Storage**:
   - Upload your data files to a Google Cloud Storage bucket.

3. **Authenticate with Google Cloud**:
   - Use the Google Cloud SDK to authenticate:
     ```bash
     gcloud auth login
     gcloud config set project <your-project-id>
     ```

4. **Build and Push Docker Image to Google Container Registry**:
   - Tag your Docker image:
     ```bash
     docker tag polysemy-annotation-app gcr.io/<your-project-id>/polysemy-annotation-app
     ```
   - Push the Docker image:
     ```bash
     docker push gcr.io/<your-project-id>/polysemy-annotation-app
     ```

5. **Deploy to Cloud Run**:
   - Deploy the application:
     ```bash
     gcloud run deploy polysemy-annotation-app --image gcr.io/<your-project-id>/polysemy-annotation-app --platform managed --region <your-region> --allow-unauthenticated
     ```

## File Descriptions

- **`app/main.py`**: The main application file that handles user interface and navigation.
- **`app/database.py`**: Contains the `Database` class for interacting with the PostgreSQL database.
- **`app/auth.py`**: Handles user authentication, including registration and login.
- **`Dockerfile`**: Defines the Docker image for the application.
- **`requirements.txt`**: Lists the Python dependencies for the application.
- **`schema.sql`**: SQL script to set up the database schema.

## Requirements

The application requires the following Python packages, as specified in `requirements.txt`:

- `streamlit==1.25.0`
- `psycopg2-binary==2.9.10`
- `bcrypt==4.0.0`
- `google-cloud-storage==2.18.2`
- `pandas==2.2.3`
- `streamlit-echarts==0.4.0`

## License

This project is licensed under the MIT License. See the LICENSE file for details.
