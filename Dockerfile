# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents
COPY . .

# Expose port 8080 (the default port for Streamlit)
EXPOSE 8080

# Command to run the Streamlit app
CMD ["streamlit", "run", "web.py", "--server.port=8080", "--server.address=0.0.0.0"]