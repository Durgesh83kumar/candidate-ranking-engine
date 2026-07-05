# Use a lightweight official Python 3.10 runtime
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable immediate output buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure Streamlit defaults for Hugging Face Spaces
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install essential system dependencies (libgomp1 is required by FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up a secure, non-root user with UID 1000 for Hugging Face compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Set up working directory inside user home
WORKDIR $HOME/app

# Copy dependency definition and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all application source files
COPY --chown=user . .

# Create the output directory inside the user home working directory
RUN mkdir -p output

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Launch the Streamlit application
CMD ["streamlit", "run", "app.py"]
