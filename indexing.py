# initial_setup.py

import os
#from llama_index.llms.vertex import Vertex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext


# Set environment variables
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC-ASsI6zwI9UiDcR9xqEH7SyeHl2MS8HY'

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set LLM model
from llama_index.llms.gemini import Gemini
Settings.llm = Gemini(model_name="models/gemini-1.0-pro-001")

# Load documents
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(input_files=['Beacon_Agent.docx']).load_data()

# Split text into chunks
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
Settings.text_splitter = text_splitter

index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])

index.storage_context.persist(persist_dir="E:/KtAgentUsingLanggraph/indexing_beacon_docx")

print("Indexing is saved")

