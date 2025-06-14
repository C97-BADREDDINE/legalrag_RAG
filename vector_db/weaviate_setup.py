import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.exceptions import WeaviateBaseError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_collection_exists(client: weaviate.WeaviateClient):
    try:
        if not client or not client.is_connected():
             raise ConnectionError("Weaviate client is not provided or not connected.")

        collection_names = client.collections.list_all()

        if "justice" not in collection_names:
            logger.info("Creating 'justice' collection...")
            client.collections.create(
                name="justice",
                vectorizer_config=Configure.Vectorizer.text2vec_huggingface(
                    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                ),
            )

            logger.info("✅ 'justice' collection created successfully.")
        else:
            logger.info("✅ 'justice' collection already exists.")

    except WeaviateBaseError as e:
        logger.error(f"Weaviate error during collection check/creation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during collection check/creation: {e}")
        raise
