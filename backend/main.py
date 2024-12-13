import os
import time
import requests
import json
import logging
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File
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

class SearchNotesRequest(BaseModel):
    query: str

class SearchNotesResponse(BaseModel):
    notes: List[Dict]

class NotesResponse(BaseModel):
    notes: List[Dict]
    total: int
class SummaryRequest(BaseModel):
    document_name: Optional[str] = None
    chapter_query: Optional[str] = None
    max_tokens: Optional[int] = 1000

# Text Splitter for embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    length_function=len,
)

# Utility Functions
def upload_to_s3(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)
    s3_client.upload_file(filename, bucket_name, filename)
    os.remove(filename)

def extract_content(file_path):
    elements = partition(filename=file_path)
    return convert_to_dict(elements)

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

def extract_chapter_content(matches: List[dict], chapter_query: str) -> str:
    """
    Extract relevant content for a specific chapter or topic from Pinecone matches.
    """
    chapter_content = []
    for match in matches:
        text = match["metadata"]["text"]
        chapter_indicators = [
            f"Chapter {chapter_query}",
            f"CHAPTER {chapter_query}",
            chapter_query.title(),
            chapter_query.upper()
        ]
        
        if any(indicator in text for indicator in chapter_indicators):
            chapter_content.append(text)
    
    if not chapter_content:
        chapter_content = [match["metadata"]["text"] for match in matches]
    
    return "\n".join(chapter_content)

async def generate_chapter_summary(content: str, chapter_query: str) -> str:
    """
    Generate a summary for the specific chapter or topic.
    """
    try:
        prompt = f"""
        Generate a comprehensive summary for the following content related to {chapter_query}.
        Focus on the main concepts, key points, and important details.
        
        Content:
        {content}
        
        Please structure the summary with these sections:
        1. Overview
        2. Key Concepts
        3. Important Details
        4. Main Takeaways
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant specializing in creating structured summaries of technical content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

async def process_uploaded_document(file):
    with open(file.filename, "wb") as f:
        f.write(await file.read())
    content = extract_content(file.filename)
    os.remove(file.filename)
    return " ".join([elem.get("text", "") for elem in content])

def get_relevant_passages(query, top_k=5):
    query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [result["metadata"]["text"] for result in results["matches"]]

# API Endpoints
@app.post("/summarize-pdf/")
async def summarize_pdf(
    file: Optional[UploadFile] = None,
    document_name: Optional[str] = None,
    topic: Optional[str] = None,
    chapter: Optional[str] = None
):
    try:
        logger.info(f"Summarize PDF request - document: {document_name}, topic: {topic}, chapter: {chapter}")
        
        if file:
            content = await process_uploaded_document(file)
            logger.info(f"Processed uploaded file with content length: {len(content)}")
        elif document_name:
            # For stored documents, use Pinecone to fetch relevant content
            query = chapter if chapter else topic if topic else document_name
            logger.info(f"Using query: {query} for document: {document_name}")
            
            # Create embedding for query
            embeddings = OpenAIEmbeddings()
            query_embedding = embeddings.embed_query(query)
            logger.info("Generated query embedding")
            
            # Query Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                filter={"file_name": document_name}
            )
            logger.info(f"Found {len(results['matches'])} matches in Pinecone")
            
            if not results["matches"]:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant content found for the specified chapter/topic"
                )
            
            # Extract content based on chapter or get all content
            if chapter:
                content = extract_chapter_content(results["matches"], chapter)
                logger.info(f"Extracted chapter content length: {len(content)}")
            else:
                content = " ".join([res["metadata"]["text"] for res in results["matches"]])
                logger.info(f"Extracted general content length: {len(content)}")
            
            if not content.strip():
                raise HTTPException(
                    status_code=404,
                    detail="No content could be extracted from the document"
                )
        else:
            raise HTTPException(status_code=400, detail="Provide a file or a stored document name.")

        # Generate summary
        logger.info("Generating summary...")
        summary = await generate_chapter_summary(content, chapter or topic or "document")
        logger.info("Summary generated successfully")

        return {
            "summary": summary,
            "document": document_name if document_name else file.filename if file else None,
            "chapter": chapter,
            "topic": topic
        }
    except Exception as e:
        logger.error(f"Error in summarize_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error summarizing PDF: {str(e)}")

@app.post("/summarize-chapter/")
async def summarize_chapter(request: SummaryRequest):
    try:
        logger.info(f"Summarize chapter request - document: {request.document_name}, query: {request.chapter_query}")
        
        if not request.document_name or not request.chapter_query:
            raise HTTPException(
                status_code=400,
                detail="Both document_name and chapter_query are required"
            )

        # Create embedding for query
        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_query(request.chapter_query)
        logger.info("Generated query embedding")
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"file_name": request.document_name}
        )
        logger.info(f"Found {len(results['matches'])} matches in Pinecone")
        
        if not results["matches"]:
            raise HTTPException(
                status_code=404,
                detail="No relevant content found for the specified chapter/topic"
            )
        
        # Extract chapter content
        content = extract_chapter_content(results["matches"], request.chapter_query)
        if not content.strip():
            raise HTTPException(
                status_code=404,
                detail="No content could be extracted for the specified chapter"
            )
        logger.info(f"Extracted content length: {len(content)}")
        
        # Generate summary
        logger.info("Generating summary...")
        summary = await generate_chapter_summary(content, request.chapter_query)
        logger.info("Summary generated successfully")
        
        return {
            "summary": summary,
            "document": request.document_name,
            "chapter_query": request.chapter_query
        }
    except Exception as e:
        logger.error(f"Error in summarize_chapter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa-pdf/")
async def qa_pdf(model: QAModel):
    try:
        logger.debug(f"Received query: {model.query}, context_id: {model.context_id}")
        
        # If context_id is provided, search in specific document
        if model.context_id:
            query_embedding = OpenAIEmbeddings().embed_query(model.query)
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                filter={"file_name": model.context_id}
            )
            
            if not results["matches"]:
                return {"answer": "No relevant information found in the selected document."}
            
            context = " ".join([res["metadata"]["text"] for res in results["matches"]])
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Answer the question based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\nQuestion: {model.query}"},
                ]
            )
        else:
            # For generic queries without document context, use both SerpAPI and GPT-4
            try:
                # First, get relevant web search results using SerpAPI
                search = GoogleSearch({
                    "q": model.query,
                    "api_key": SERPAPI_KEY
                })
                search_results = search.get_dict()
                
                # Extract relevant information from search results
                web_context = ""
                if "organic_results" in search_results:
                    for result in search_results["organic_results"][:3]:  # Take top 3 results
                        web_context += f"\nTitle: {result.get('title', '')}\n"
                        web_context += f"Snippet: {result.get('snippet', '')}\n"

                # Combine web search results with GPT-4
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable quantum computing tutor. Use the provided web search context and your knowledge to answer the question comprehensively."},
                        {"role": "user", "content": f"Web Search Context: {web_context}\nQuestion: {model.query}"}
                    ]
                )

            except Exception as e:
                logger.error(f"Error in generic search: {e}")
                # Fallback to just GPT-4 if SerpAPI fails
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable quantum computing tutor. Answer the question comprehensively."},
                        {"role": "user", "content": model.query}
                    ]
                )

        # Get relevant papers from arXiv
        try:
            papers_response = requests.post(
                f"{BASE_URL}/fetch-research-papers/",
                json={"topic": model.query}
            )
            papers = papers_response.json().get("papers", []) if papers_response.status_code == 200 else []
        except Exception as e:
            logger.warning(f"Failed to fetch papers: {e}")
            papers = []

        return {
            "answer": response.choices[0].message.content,
            "papers": papers,
            "has_document_context": bool(model.context_id)
        }

    except Exception as e:
        logger.exception(f"Error in qa_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Q&A: {str(e)}")

@app.get("/embeddings-stats/")
async def get_embeddings_stats():
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": stats.namespaces
        }
    except Exception as e:
        logger.error(f"Error getting embeddings stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        return {"answer": response.choices[0].message.content}
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

@app.get("/search-notes/")
async def search_notes(query: str):
    """
    Search through stored notes using the provided query.
    """
    try:
        # Get all notes
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="notes/")
        matching_notes = []
        
        for obj in response.get("Contents", []):
            note_file = obj["Key"]
            local_filename = note_file.split("/")[-1]
            s3_client.download_file(bucket_name, note_file, local_filename)
            
            with open(local_filename, "r") as f:
                note = json.load(f)
                # Search in both title and content
                if (query.lower() in note["title"].lower() or 
                    query.lower() in note["content"].lower()):
                    matching_notes.append(note)
            
            os.remove(local_filename)
        
        return {"notes": matching_notes}
    except Exception as e:
        logger.error(f"Error searching notes: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching notes: {str(e)}")

@app.get("/get-notes-metadata/")
async def get_notes_metadata():
    """
    Get metadata about stored notes (count, dates, etc.).
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="notes/")
        total_notes = len(response.get("Contents", []))
        return {"total": total_notes}
    except Exception as e:
        logger.error(f"Error getting notes metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhance existing save_note endpoint
@app.post("/save-note/")
async def save_note(model: SaveNoteModel):
    try:
        timestamp = datetime.now().isoformat()
        note_id = str(int(time.time()))
        note = {
            "id": note_id,
            "title": model.title,
            "content": model.content,
            "created_at": timestamp,
            "updated_at": timestamp
        }
        
        # Save to S3
        upload_to_s3(note, f"notes/{note_id}.json")
        
        # Create embedding for the note content
        try:
            note_embedding = get_embedding(f"{model.title} {model.content}")
            metadata = {
                "text": f"{model.title}\n{model.content}",
                "title": model.title,
                "note_id": note_id,
                "type": "note"
            }
            
            # Store in Pinecone for semantic search
            index.upsert(
                vectors=[(f"note_{note_id}", note_embedding, metadata)]
            )
        except Exception as e:
            logger.warning(f"Failed to create embedding for note: {e}")
        
        return {
            "message": "Note saved successfully.",
            "note_id": note_id,
            "created_at": timestamp
        }
    except Exception as e:
        logger.error(f"Error saving note: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving note: {str(e)}")


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