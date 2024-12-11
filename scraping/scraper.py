import os
import boto3
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
import time
import random
from pinecone.exceptions import PineconeProtocolError
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Initialize AWS S3 client
def init_s3():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    bucket_name = os.getenv("AWS_BUCKET")
    return s3, bucket_name

# Initialize Pinecone client and index
def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-1"),
        )
    return pc.Index(index_name)

# Extract content from various file types using Unstructured.io
def extract_content(file_path):
    elements = partition(filename=file_path)
    return convert_to_dict(elements)

# Scrape IBM Quantum Blog for articles
def scrape_ibm_blog():
    url = "https://www.ibm.com/quantum/blog"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []
    for article in soup.select(".ibm--card"):
        title = article.select_one(".ibm--card__title").text.strip()
        link = article.select_one("a")["href"]
        summary = article.select_one(".ibm--card__content").text.strip()
        data.append({"title": title, "link": link, "summary": summary, "source": "IBM Quantum Blog"})
    return data

def safe_upsert(index, vectors, max_retries=5):
    """
    Safely upsert vectors into Pinecone with retry logic.

    Args:
        index: The Pinecone index object.
        vectors: List of vectors to upsert.
        max_retries: Maximum number of retries before failing.

    Returns:
        None
    """
    retries = 0
    while retries < max_retries:
        try:
            index.upsert(vectors)
            return  # Success
        except PineconeProtocolError as e:
            print(f"Retrying upsert due to PineconeProtocolError: {e}")
            retries += 1
            time.sleep(2 ** retries + random.uniform(0, 1))  # Exponential backoff with jitter
    raise Exception("Max retries reached; upsert failed.")

# Scrape Microsoft Quantum Blog for articles
def scrape_microsoft_blog():
    url = "https://azure.microsoft.com/en-us/blog/quantum/content-type/news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []
    for article in soup.select(".item-container"):
        title = article.select_one("h3").text.strip()
        link = "https://azure.microsoft.com" + article.select_one("a")["href"]
        summary = article.select_one(".desc").text.strip() if article.select_one(".desc") else ""
        data.append({"title": title, "link": link, "summary": summary, "source": "Microsoft Quantum Blog"})
    return data

# Scrape Quantum StackExchange for questions and answers
def scrape_stackexchange():
    url = "https://quantumcomputing.stackexchange.com/questions"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []
    for question in soup.select(".question-summary"):
        title = question.select_one(".question-hyperlink").text.strip()
        link = "https://quantumcomputing.stackexchange.com" + question.select_one(".question-hyperlink")["href"]
        excerpt = question.select_one(".excerpt").text.strip()
        data.append({"title": title, "link": link, "excerpt": excerpt, "source": "Quantum StackExchange"})
    return data

# Upload JSON data to S3 bucket
def upload_to_s3(data, prefix, s3, bucket_name):
    filename = f"{prefix}.json"
    with open(filename, "w") as f:
        json.dump(data, f)
    s3.upload_file(filename, bucket_name, filename)
    os.remove(filename)
    print(f"Uploaded {filename} to S3.")

# Split text into chunks using LangChain's RecursiveCharacterTextSplitter
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunk size to generate more embeddings
        chunk_overlap=200,  # Overlap between chunks to maintain context
        length_function=len,
    )
    return text_splitter.split_text(text)

# Generate embeddings using OpenAI's API and store them in Pinecone
def generate_embeddings():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    s3, bucket_name = init_s3()
    pc_index = init_pinecone()
    embeddings_model = OpenAIEmbeddings()

    response = s3.list_objects_v2(Bucket=bucket_name)
    for obj in response.get("Contents", []):
        key = obj["Key"]
        local_filename = key.split("/")[-1]

        s3.download_file(bucket_name, key, local_filename)
        try:
            if local_filename.endswith(".json"):
                with open(local_filename, "r") as f:
                    data = json.load(f)
                for item in data:
                    text_content = f"{item.get('title', '')} {item.get('summary', '')} {item.get('excerpt', '')}"
                    chunks = split_text(text_content)
                    print(f"Number of chunks for {local_filename}: {len(chunks)}")
                    for i, chunk in enumerate(chunks):
                        embedding_vector = embeddings_model.embed_query(chunk)
                        metadata = {
                            "text": chunk,
                            "file_name": local_filename,  # Add file name
                            "document_title": item.get("title", ""),  # Add title if available
                        }
                        safe_upsert(pc_index, [(f"{local_filename}_chunk_{i}", embedding_vector, metadata)])
            else:
                content_elements = extract_content(local_filename)
                text_content = " ".join([elem.get("text", "") for elem in content_elements])
                chunks = split_text(text_content)
                print(f"Number of chunks for {local_filename}: {len(chunks)}")
                for i, chunk in enumerate(chunks):
                    embedding_vector = embeddings_model.embed_query(chunk)
                    metadata = {
                        "text": chunk,
                        "file_name": local_filename,  # Add file name
                    }
                    safe_upsert(pc_index, [(f"{local_filename}_chunk_{i}", embedding_vector, metadata)])
        except Exception as e:
            print(f"Error processing {local_filename}: {e}")
        finally:
            os.remove(local_filename)

# Main function to orchestrate the pipeline process
def main():
    
     # Initialize S3 client and bucket name
    
     s3_client , bucket_name_value= init_s3()

     # Scrape data from IBM Quantum Blog
    
     print("Scraping IBM Quantum Blog...")
     
     ibm_data_scraped= scrape_ibm_blog()
     
     upload_to_s3(ibm_data_scraped , "ibm_quantum_blog", s3_client , bucket_name_value)

     # Scrape data from Microsoft Quantum Blog
    
     print("Scraping Microsoft Quantum Blog...")
     
     microsoft_data_scraped= scrape_microsoft_blog()
     
     upload_to_s3(microsoft_data_scraped , "microsoft_quantum_blog", s3_client , bucket_name_value)

     # Scrape data from Quantum StackExchange
    
     print("Scraping Quantum StackExchange...")
     
     stackexchange_data_scraped= scrape_stackexchange()
     
     upload_to_s3(stackexchange_data_scraped , "quantum_stackexchange", s3_client , bucket_name_value)

     # Generate embeddings for all collected data
    
     print("Generating embeddings...")
     
     generate_embeddings()

if __name__ == "__main__":
    
   main()