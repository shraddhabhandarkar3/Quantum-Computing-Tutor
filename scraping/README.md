# Quantum Pipeline

This project scrapes data from three websites, processes PDF content from an S3 bucket, and generates vector embeddings using OpenAI and Pinecone.

## Requirements

- Python 3.10.15
- Poetry

## Installation

1. Clone the repository.
2. Run `poetry install` to set up the environment.
3. Create a `.env` file with the following variables:

```env
AWS_ACCESS_KEY_ID=<Your AWS Access Key>
AWS_SECRET_ACCESS_KEY=<Your AWS Secret Key>
AWS_BUCKET=<Your S3 Bucket Name>
PINECONE_API_KEY=<Your Pinecone API Key>
PINECONE_INDEX_NAME=<Your Pinecone Index Name>
OPENAI_API_KEY=<Your OpenAI API Key>
