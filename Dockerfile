# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 to match Cloud Run’s default
EXPOSE 8080

# Command to run the app from the 'app' directory
CMD ["streamlit", "run", "app/main.py", "--server.port=8080", "--server.headless=true"]
