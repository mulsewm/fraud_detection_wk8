# Use an official Python runtime as base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install flask joblib pandas scikit-learn

# Expose port 5000
EXPOSE 8000

# Run Flask application
CMD ["python", "serve_model.py"]
