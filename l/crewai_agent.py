import os
import logging
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
from pydantic import BaseModel
from typing import List
import litellm
from file import load_documents_from_directory
from llm_config import LlamaStackLLM, llm_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress litellm debug output
litellm.set_verbose = False

# Configure litellm for Ollama
litellm.api_base = "http://183.83.216.62:8321"
litellm.model = "meta-llama/Llama-3.2-3B-Instruct"

# Load environment variables
load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data")
DB_NAME = os.getenv("DATABASE_NAME", "database_llama")
CONN_STR = os.getenv("POSTGRESQL_CONN_STR", "postgresql://postgres:Techmo@localhost:5433")

# Initialize PGVectorStore
try:
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
    logger.info("PGVectorStore initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing PGVectorStore: {str(e)}")
    raise

# Define Agents
try:
    document_processor = Agent(
        role="Document Processor",
        goal="Load and index documents into the vector store for efficient retrieval",
        backstory="You're an expert in document management, ensuring files are properly processed and indexed for quick access.",
        llm=LlamaStackLLM(client=llm_client, model_id="meta-llama/Llama-3.2-3B-Instruct"),
        verbose=True
    )

    query_responder = Agent(
        role="Query Responder",
        goal="Answer user queries accurately based on indexed documents",
        backstory="You're a knowledgeable assistant skilled at retrieving and synthesizing information from document indices.",
        llm=LlamaStackLLM(client=llm_client, model_id="meta-llama/Llama-3.2-3B-Instruct"),
        verbose=True
    )
    logger.info("Agents created successfully.")
except Exception as e:
    logger.error(f"Error creating agents: {str(e)}")
    raise

# Define Tasks
class DocumentProcessTask(BaseModel):
    directory_path: str = UPLOAD_DIR

class QueryTask(BaseModel):
    query: str

def process_documents_task(directory_path: str = UPLOAD_DIR):
    try:
        documents = load_documents_from_directory(directory_path)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            show_progress=True
        )
        logger.info("Documents indexed successfully.")
        return "Documents processed and indexed successfully."
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return f"Error processing documents: {str(e)}"

document_task = Task(
    description="Load and index documents from the specified directory.",
    expected_output="A confirmation message indicating successful document indexing.",
    agent=document_processor,
    execute=lambda task_inputs: process_documents_task(task_inputs.directory_path),
    inputs=DocumentProcessTask(directory_path=UPLOAD_DIR)
)

def execute_query_task(query: str):
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        query_engine = index.as_query_engine(llm=LlamaStackLLM(client=llm_client, model_id="meta-llama/Llama-3.2-3B-Instruct"))
        response = query_engine.query(query)
        logger.info(f"Query executed successfully: {query}")
        return str(response)
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return f"Error executing query: {str(e)}"

query_task = Task(
    description="Answer the user's query using the indexed documents.",
    expected_output="A clear and accurate response to the user's query.",
    agent=query_responder,
    execute=lambda task_inputs: execute_query_task(task_inputs.query),
    inputs=QueryTask(query="")
)

# Create Crew
try:
    crew = Crew(
        agents=[document_processor, query_responder],
        tasks=[document_task, query_task],
        process=Process.sequential,
        verbose=True
    )
    logger.info("Crew initialized successfully.")
except Exception as e:
    logger.error(f"Error creating crew: {str(e)}")
    raise

# Function to run the crew for a specific query
def run_crew(query: str = None):
    try:
        if query:
            query_task.inputs = QueryTask(query=query)
            result = crew.kickoff()
            logger.info(f"Crew executed successfully for query: {query}")
            return result
        else:
            result = document_task.execute()
            logger.info("Document processing completed.")
            return result
    except Exception as e:
        logger.error(f"Error running crew: {str(e)}")
        return f"Error running crew: {str(e)}"

if __name__ == "__main__":
    try:
        # Example: Process documents and answer a query
        result = run_crew(query="What are the key points in the uploaded documents?")
        print(result)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")