import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import weaviate
import sys
from weaviate.classes.init import Auth
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from vector_db.weaviate_setup import ensure_collection_exists

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for services
weaviate_client = None
file_processing_service = None
retriever_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global weaviate_client, file_processing_service, retriever_service
    logger.info("Application startup...")
    try:
        # 1. Initialize Weaviate Cloud Client
        logger.info("Connecting to Weaviate Cloud...")
        
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_key = os.getenv("WEAVIATE_API_KEY")
        
        if not weaviate_url or not weaviate_key:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY environment variables must be set.")
        
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_key),
        )
        
        if not weaviate_client.is_ready():
             raise ConnectionError("Failed to connect to Weaviate Cloud.")
        logger.info(f"✅ Connected to Weaviate Cloud at {weaviate_url}")

        # 2. Ensure Weaviate Collection Exists
        ensure_collection_exists(weaviate_client) # Pass the initialized client

        # 3. Initialize Services with the Client
        logger.info("Initializing services...")
        # Note: You'll need to import FileProcessingService and Retriever when they're implemented
        # file_processing_service = FileProcessingService(client=weaviate_client)
        # await file_processing_service.initialize() # Initialize internal components (e.g., model pool)

        # retriever_service = Retriever(client=weaviate_client)
        logger.info("✅ Services initialized successfully.")

        yield # Application runs here

    except Exception as e:
         logger.error(f"❌ Application startup failed: {e}", exc_info=True)
         # Optionally re-raise or handle specific startup errors
         raise # Raising error will prevent FastAPI from starting properly if critical components fail
    finally:
        # --- Shutdown ---
        logger.info("Application shutdown...")
        # 1. Close Services (releasing internal resources like model pools)
        if file_processing_service:
            await file_processing_service.close()

        # 2. Close Weaviate Client
        if weaviate_client and weaviate_client.is_connected():
            weaviate_client.close()
            logger.info("✅ Weaviate client closed.")
        logger.info("✅ Application shutdown complete.")


# Create FastAPI app instance with the lifespan manager
app = FastAPI(
    title="File Processing and Chat API",
    description="API for uploading files to Weaviate and chatting with the indexed data.",
    version="1.1.0", 
    lifespan=lifespan # Register the lifespan context manager
)

# Add your API routes here
@app.get("/")
async def root():
    return {"message": "Legal RAG API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client.is_connected() if weaviate_client else False
    }

# Add this main block to run the app when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
