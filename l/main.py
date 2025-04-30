import os
import psycopg2
from fastapi import FastAPI, UploadFile, File
from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import make_url
from pydantic import BaseModel
from file import load_documents_from_directory
from llm_config import LlamaStackLLM, llm_client
from dotenv import load_dotenv
import logging
import litellm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure litellm for Ollama
litellm.set_verbose = False
litellm.api_base = "http://183.83.216.62:8321"
litellm.model = "meta-llama/Llama-3.2-3B-Instruct"

# Load environment variables
load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data")
DB_NAME = os.getenv("DATABASE_NAME", "database_llama")
CONN_STR = os.getenv("POSTGRESQL_CONN_STR", "postgresql://postgres:Techmo@localhost:5433")

# FastAPI App
app = FastAPI()

# Set global embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# --- Ensure Database Exists ---
def ensure_database_exists():
    try:
        conn = psycopg2.connect(CONN_STR)
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"Database '{DB_NAME}' created successfully.")
        else:
            logger.info(f"Database '{DB_NAME}' already exists.")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error ensuring database exists: {str(e)}")
        raise

ensure_database_exists()

# Initialize LLM and PGVectorStore
model_id = "meta-llama/Llama-3.2-3B-Instruct"
llm = LlamaStackLLM(client=llm_client, model_id=model_id)

pg_url = make_url(CONN_STR)
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=pg_url.host,
    password=pg_url.password,
    port=pg_url.port,
    user=pg_url.username,
    table_name="files_data",
    embed_dim=768,
    hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40, "hnsw_dist_method": "vector_cosine_ops"}
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Upload Endpoint ---
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not any(file.filename.endswith(ext) for ext in [".pdf", ".docx", ".txt"]):
        return {"error": "Unsupported file format."}

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        documents = load_documents_from_directory(UPLOAD_DIR)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            show_progress=True
        )
        logger.info(f"File '{file.filename}' uploaded and indexed successfully.")
        return {"message": f"File '{file.filename}' uploaded and indexed successfully."}
    except Exception as e:
        logger.error(f"Failed to process file: {str(e)}")
        return {"error": f"Failed to process file: {str(e)}"}

# --- Query Endpoint ---
class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_index(request: QueryRequest):
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(request.query)
        logger.info(f"Query executed successfully: {request.query}")
        return {"response": str(response)}
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return {"error": f"Query failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)