import os
import time
import requests
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dict
from semantic_router.encoders import OpenAIEncoder
from pinecone import Pinecone, ServerlessSpec
import boto3
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Quantum Computing GPT Backend",
    description="Backend for Quantum Computing GPT functionalities.",
    version="1.0",
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI Encoder
encoder = OpenAIEncoder(name="text-embedding-3-small")

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY is not set in environment variables.")
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
dims = len(encoder(["some random text"])[0])
index_name = os.getenv("PINECONE_INDEX_NAME")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=dims,
        metric="dotproduct",
        spec=spec,
    )

while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)
index.describe_index_stats()

# Initialize AWS S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
bucket_name = os.getenv("AWS_BUCKET")
if not bucket_name:
    raise ValueError("AWS_BUCKET is not set in environment variables.")

# Pydantic Models
class QAModel(BaseModel):
    query: str
    context_id: Optional[str] = None

class GenericQAModel(BaseModel):
    query: str

class ResearchPaperModel(BaseModel):
    topic: str

class SaveNoteModel(BaseModel):
    title: str
    content: str

class DeleteNoteModel(BaseModel):
    note_id: str

# Text Splitter for embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    length_function=len,
)

# Utils: Upload JSON to S3
def upload_to_s3(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)
    s3_client.upload_file(filename, bucket_name, filename)
    os.remove(filename)

# Utils: Extract content from PDF
def extract_content(file_path):
    elements = partition(filename=file_path)
    return convert_to_dict(elements)

# Utils: Generate embeddings
def generate_embeddings(content, metadata=None):
    embeddings_model = OpenAIEmbeddings()
    chunks = text_splitter.split_text(content)
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings_model.embed_query(chunk)
        metadata = metadata or {}
        metadata["text"] = chunk
        vectors.append((f"{metadata.get('id', 'chunk')}_{i}", embedding, metadata))
    index.upsert(vectors)
    return len(vectors)

# Endpoints
@app.post("/summarize-pdf/")
async def summarize_pdf(file: Optional[UploadFile] = None, document_name: Optional[str] = None, topic: Optional[str] = None, chapter: Optional[str] = None):
    try:
        if file:
            content = await process_uploaded_document(file)
        elif document_name:
            content = fetch_document_from_pinecone(document_name)
        else:
            raise HTTPException(status_code=400, detail="Provide a file or a stored document name.")

        if chapter:
            content = extract_chapter(content, chapter)

        summary = generate_summary(content, topic)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing PDF: {str(e)}")

async def process_uploaded_document(file):
    with open(file.filename, "wb") as f:
        f.write(await file.read())
    content = extract_content(file.filename)
    os.remove(file.filename)
    return " ".join([elem.get("text", "") for elem in content])

def fetch_document_from_pinecone(document_name):
    results = index.query(
        vector=[0] * dims,
        top_k=10000,
        include_metadata=True,
        filter={"file_name": document_name}
    )
    return " ".join([res["metadata"]["text"] for res in results["matches"]])

def extract_chapter(content, chapter):
    # Implement chapter extraction logic here
    # This is a placeholder implementation
    return content

def generate_summary(content, topic=None):
    prompt = f"Summarize the following text about {topic}:" if topic else "Summarize the following text:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
    )
    return response.choices[0].message.content

@app.post("/qa-pdf/")
async def qa_pdf(model: QAModel):
    try:
        logger.debug(f"Received query: {model.query}, context_id: {model.context_id}")
        if not model.context_id:
            raise HTTPException(status_code=400, detail="Document name (context_id) is required.")
        
        query_embedding = OpenAIEmbeddings().embed_query(model.query)
        logger.debug(f"Generated query embedding")
        
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"file_name": model.context_id}
        )
        logger.debug(f"Pinecone query results: {results}")
        
        if not results["matches"]:
            return {"answer": "No relevant information found in the selected document."}
        
        context = " ".join([res["metadata"]["text"] for res in results["matches"]])
        logger.debug(f"Generated context: {context[:100]}...")  # Log first 100 characters
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer the question based on the provided context."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {model.query}"},
            ]
        )
        answer = response.choices[0].message.content
        logger.debug(f"Generated answer: {answer[:100]}...")  # Log first 100 characters
        
        return {"answer": answer}
    except Exception as e:
        logger.exception(f"Error in qa_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Q&A: {str(e)}")

@app.post("/generic-qa/")
async def generic_qa(model: GenericQAModel):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer the following question as accurately as possible."},
                {"role": "user", "content": model.query},
            ]
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generic Q&A: {e}")

@app.post("/fetch-research-papers/")
async def fetch_research_papers(model: ResearchPaperModel):
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{model.topic}&start=0&max_results=5"
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch research papers.")
        papers = []
        entries = response.text.split("<entry>")
        for entry in entries[1:]:
            title_start = entry.find("<title>") + 7
            title_end = entry.find("</title>")
            link_start = entry.find("<id>") + 4
            link_end = entry.find("</id>")
            papers.append({"title": entry[title_start:title_end], "link": entry[link_start:link_end]})
        return {"papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching papers: {e}")

@app.post("/save-note/")
async def save_note(model: SaveNoteModel):
    try:
        note_id = str(int(time.time()))
        note = {"id": note_id, "title": model.title, "content": model.content}
        upload_to_s3(note, f"notes/{note_id}.json")
        return {"message": "Note saved successfully.", "note_id": note_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving note: {e}")

@app.get("/get-notes/")
async def get_notes():
    try:
        notes = []
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="notes/")
        for obj in response.get("Contents", []):
            note_file = obj["Key"]
            local_filename = note_file.split("/")[-1]
            s3_client.download_file(bucket_name, note_file, local_filename)
            with open(local_filename, "r") as f:
                notes.append(json.load(f))
            os.remove(local_filename)
        return {"notes": notes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching notes: {e}")

@app.post("/delete-note/")
async def delete_note(model: DeleteNoteModel):
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=f"notes/{model.note_id}.json")
        return {"message": "Note deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting note: {e}")

@app.get("/publications/")
async def get_publications():
    try:
        results = index.query(
            vector=[0] * dims,
            top_k=10000,
            include_metadata=True
        )
        documents = {}
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            file_name = metadata.get("file_name")
            document_title = metadata.get("document_title")
            if file_name and file_name not in documents:
                documents[file_name] = {
                    "file_name": file_name,
                    "document_title": document_title or file_name
                }
        
        if "978-3-030-61601-4.pdf" in documents:
            documents["978-3-030-61601-4.pdf"]["document_title"] = "Quantum Computing for the Quantum Curious"
        
        return {"publications": list(documents.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching publications: {str(e)}")

@app.get("/check-pinecone-index/")
async def check_index():
    try:
        dummy_vector = [0] * dims
        results = index.query(vector=dummy_vector, top_k=1000, include_metadata=True)
        return {
            "matches": len(results.get("matches", [])),
            "metadata": [match.get("metadata", {}) for match in results["matches"]],
        }
    except Exception as e:
        logger.exception(f"Error checking Pinecone index: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking Pinecone index: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
