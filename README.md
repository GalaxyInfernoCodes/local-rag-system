# Local RAG System Setup

This repository contains a Local Retrieval-Augmented Generation (RAG) System designed as a learning project. The goal is to build a simplified version of a RAG system from scratch to understand the underlying components and their interactions. It is not intended to be a fully optimized or production-ready solution but serves as an educational tool to explore concepts like embedding, vector storage, and query handling using a build-from-scratch approach.

## Prerequisites

- Docker and Docker Compose installed on your machine.
- An OpenAI API key for accessing OpenAI services as the model used for embeddings and generations is currently OpenAI-based.

## Environment Variables

Before starting, ensure you have set the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or include this variable in a local `.env` file.

Replace `your-openai-api-key` with your actual OpenAI API key.

## Setup Instructions

### Step 1: Build and Run Docker Containers

1. Navigate to the project directory:
2. Build and start the Docker containers:
   ```bash
   docker compose up -d
   ```

A tutorial on how the database is set up and explanations for the settings can be found [here](https://sarahglasmacher.com/how-to-pgvector-docker-local-vector-database/).

### Step 2: Install Python Requirements

Set up a local virtual environment and install the Python requirements using `uv`:
   ```bash
   uv install
   ```

This command will create a virtual environment and install all dependencies specified in the `pyproject.toml` file.

Alternatively, you can use another tool like `poetry` to install the needed libraries found as dependencies in the `pyproject.toml` file.

### Step 3: Fill the RAG Database

1. Run the database setup script:
   ```bash
   python rag_setup.py
   ```
   This script will populate the RAG database with the necessary data.

### Step 4: Start the Frontend

1. Run the frontend script to start the Gradio application:
   ```bash
   python frontend.py
   ```

   This will start a local server where you can interact with the chat application. Please keep in mind that every single query picks out the most similar embedding from the database and generates an answer, even if no embedding is relevant to the query. 

## Usage

Once the setup is complete, open your browser and navigate to the local server URL provided by Gradio to start using the chat application.
