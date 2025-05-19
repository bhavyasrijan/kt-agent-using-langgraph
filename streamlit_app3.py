import tempfile
import traceback

import streamlit as st
import base64
import os
from streamlit_float import *
from audio_recorder_streamlit import audio_recorder
from retriever import get_rag_chain,get_embeddings,get_local_llm
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from io import BytesIO
from IPython.display import Audio, display
from podcastfy.client import generate_podcast
import vertexai 
from google.cloud import aiplatform
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

os.environ['GOOGLE_APPLICATION_CREDENTIALS']= 'my_service_account2.json'
os.environ['GEMINI_API_KEY']='AIzaSyC-ASsI6zwI9UiDcR9xqEH7SyeHl2MS8HY'

import vertexai 

PROJECT_ID = "sixth-module-394805"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")





float_init()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content_eng": "Hello! I am your personal CTE KT assistant. Which project do you need help with?"}]
    if "stop_audio" not in st.session_state:
        st.session_state.stop_audio = False
    if "user_query_eng" not in st.session_state:
        st.session_state.user_query_eng = ""
    if "widget" not in st.session_state:
        st.session_state.widget = ""

def submit():
    st.session_state.user_query_eng = st.session_state.widget
    st.session_state.widget = ''  # Clear the input after submission

def autoplay_audio(file_path:str):
    with open(file_path,'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')

    md = f"""<audio autoplay>                                       
    <source src= "data: audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

    st.markdown(md,unsafe_allow_html=True)

def pretty_print_docs(docs):
    """Displays loaded documents in a structured format"""
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def generate_embedding_and_vector(texts):
    """
    Creates text embeddings using Vertex AI/HuggingFace embeddings and builds a Chroma vector index

    Args:
        texts: A list of text chunks

    Returns:
        A Chroma vector index ready for similarity search
    """
    vector_index = Chroma.from_texts(texts,get_embeddings()).as_retriever()  #vertex_embeddings
    return vector_index

def generate_embeddings_and_vector(texts):
    """Generate vector store from texts"""
    if texts and isinstance(texts[0], str):
        # Convert strings to Document objects if needed
        texts = [Document(page_content=text) for text in texts]
    vector_index = Chroma.from_documents(
        documents=texts, 
        embedding=embedding_function,  #get_embeddings()
        persist_directory="./chroma_db"
    )
    return vector_index

def get_similar_documents(vector_index, search_query):
    """
    Finds documents semantically relevant to a query using the vector index

    Args:
        vector_index: The Chroma vector index to search within
        search_query: The user's search query

    Returns:
        A list of relevant documents
    """
    docs = vector_index.get_relevant_documents(search_query)
    return docs

def process_file(fileobj, search_query):
    """
    Loads a supported document, extracts its text content, and generates an answer to a provided query based on the document.

    Args:
        fileobj: A file-like object representing the document.
        search_query: The user's question about the document.

    Returns:
        A text string containing the answer, or "Failed to load the document" if an error occurs.
    """

    file_path = fileobj.name
    filename, file_extension = os.path.splitext(file_path)

    if file_extension == '.txt':
        loader = TextLoader(file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in documents)
        texts = text_splitter.split_text(context)

    elif file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()

        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in documents)
        texts = text_splitter.split_text(context)

    elif file_extension in ['.pptx', '.ppt']:
        loader = UnstructuredPowerPointLoader(file_path)
        documents = loader.load_and_split()

        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in documents)
        texts = text_splitter.split_text(context)

    elif file_extension in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load_and_split()

        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in documents)
        texts = text_splitter.split_text(context)

    elif file_extension == '.csv':
        loader = CSVLoader(file_path)
        documents = loader.load()
        texts = [str(p.page_content) for p in documents]

    else:
        return "Unsupported file type"

    if len(texts) > 0:
        vector_index = generate_embeddings_and_vector(texts)

        llm = get_local_llm()
        _filter = LLMChainFilter.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_filter,base_retriever = vector_index.as_retriever()
        )

        compressed_docs = compression_retriever.get_relevant_documents(search_query)
        context_text = [i.page_content for i in compressed_docs]
        response_text = get_rag_chain().invoke({"context": context_text, "question": search_query})
        
        pretty_print_docs(compressed_docs)
        return response_text

    else:
        return "Failed to load the document"

def embed_audio_streamlit(audio_file):
    """
    Embed and play an audio file in Streamlit
    
    Args:
        audio_file (str): Path to the audio file
    """
    with open(audio_file, 'rb') as audio_bytes:
        st.audio(audio_bytes.read(), format='audio/mp3')

# Initialize session state
initialize_session_state()

# Main Streamlit app
st.title("    CTE Knowledge Transfer Assistant 🤖    ")

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content_eng'])

# Floating footer container
footer_container = st.container()
with footer_container:
    col1, col2, col3 = st.columns([4, 1, 1])  # Adjusted column widths
    with col1:
        st.text_input("Type your query here...")
    with col2:
        audio_bytes = audio_recorder(text='', recording_color="#e8b62c",
    neutral_color="#6aa36f", icon_size="3x")
    with col3:
        stop_button = st.button("Stop Audio")

# Handle stop button
if stop_button:
    st.session_state.stop_audio = True

# Main processing logic
if (audio_bytes or st.session_state.user_query_eng) and not st.session_state.stop_audio:
    if audio_bytes:
        print('entering the audio bytes...')
        with st.spinner("Transcribing..."):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            # Note: Transcription function is missing - need to implement this
            # For now, I'll leave a placeholder
            transcript_eng = ""  # Replace with actual transcription logic

            if transcript_eng:
                st.session_state.messages.append({"role": "user", "content_eng": transcript_eng})
                with st.chat_message("user"):
                    st.write(transcript_eng)
                os.remove(webm_file_path)

    '''elif st.session_state.user_query_eng:
        print('entering the user input...')
        st.session_state.messages.append({"role": "user", "content_eng": st.session_state.user_query_eng})
        with st.chat_message("user"):
            st.write(st.session_state.user_query_eng)'''


#Process the assistant response
#if st.session_state.user_query_eng and st.session_state.messages[-1]["role"] == "user" and not st.session_state.stop_audio:

    

    # Sidebar for document upload and querying
st.sidebar.header("Document Query")

# File uploader with a unique key
uploaded_file = st.sidebar.file_uploader(
    "Upload your document", 
    type=['txt', 'pdf', 'pptx', 'ppt', 'docx', 'doc', 'csv'],
    key="document_upload_sidebar_unique"
)

# Text input for query with a unique key
file_query = st.sidebar.text_input(
    "Enter your query about the document:", 
    key="widget",on_change=submit
)

# Process document if both file and query are present
if uploaded_file is not None and file_query:
    try:
        # Expanded error handling and logging
        st.sidebar.info(f"Processing document: {uploaded_file.name}")
        
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Use the process_file function
        with st.spinner("Processing document..."):
            document_response = process_file(uploaded_file, file_query)
            
            # Display response in sidebar
            st.sidebar.success("Document query processed successfully!")
            
            # Display response in main area
            st.write("Document Query Response:")
            st.write(document_response)

            # Append to session messages
            st.session_state.messages.append({
                "role": "assistant",
                "content_eng": document_response
            })

    except Exception as e:
        # Detailed error handling
        st.sidebar.error(f"An error occurred: {str(e)}")
        st.sidebar.error(f"Error details: {traceback.format_exc()}")

# Process the assistant's response

        

st.session_state.user_query_eng = ""

# Reset the stop_audio flag
if st.session_state.stop_audio:
    st.session_state.stop_audio = False

# Podcast section
uploaded_podcast_file = st.file_uploader(
    "Upload your document for creating podcast", 
    type=['txt', 'pdf', 'pptx', 'ppt', 'docx', 'doc', 'csv'],
    key="document_upload_podcast"
)

if uploaded_podcast_file:
    st.write(f"Uploaded file: **{uploaded_podcast_file.name}**")
    
    # Button to generate podcast
    if st.button("Generate Podcast"):
        with st.spinner("Generating podcast audio..."):
            # Save the uploaded file to a temporary location
            file_path = uploaded_podcast_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_podcast_file.getbuffer())
            
            try:
                # Generate the podcast audio
                audio_file = generate_podcast(urls=[file_path], tts_model="geminimulti")
                
                # Embed the audio in the app
                st.write("Podcast Audio:")
                embed_audio_streamlit(audio_file)
                
                st.success("Podcast generation completed!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Float the footer container to the bottom
footer_container.float("bottom: 0rem; color: white; background-color: #333; padding: 10px;")