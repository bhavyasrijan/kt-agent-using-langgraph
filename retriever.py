from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate



os.environ["TAVILY_API_KEY"] = "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"

# LLM
local_llm = "llama3.2:1b-instruct-q4_K_M"


# LLM
#llm = ChatOllama(model=local_llm, format="json", temperature=0)


# Set environment variables
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC-ASsI6zwI9UiDcR9xqEH7SyeHl2MS8HY'

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = Settings.embed_model

def get_embeddings():
    return embed_model
 
# Set LLM model

Settings.llm = Gemini(model_name="models/gemini-1.5-pro-002",temperature=0.0)


# Rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="E:/KtAgentUsingLanggraph/indexing_beacon_docx")
 
# Load index
index = load_index_from_storage(storage_context)


retriever = index.as_retriever()
#nodes = retriever.retrieve("What is beacon?")
#print(nodes)

#=======================retreiver-grader===================

'''
# Retrieval Grader



prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = "what is beacon"
docs = retriever.retrieve(question)
doc_txt = docs[1].node.text
print("yes")
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
'''


#=============Generator======================


#Generator
# Generate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use the required number of sentences to keep the answer well detailed <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

llm = ChatOllama(model=local_llm, temperature=0)

def get_local_llm():
    return llm

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.node.text for doc in docs) 


# Chain
rag_chain = prompt | llm | StrOutputParser()



# Run
import time

question = "who is the founder and ceo of beacon"
#give me the architecture diagram for the grounding technique
#fetch the whole preamble from the codebase(large model answering)
#explain, in detail,about the prompt/preamble used to instruct the gemini model to generate the answer in the chatbot
#Tell me in detail about the whole project from start till now
#What is the tech stack used in the project, explain in detail
#What is the final cost estimate with and without translation of the beacon project?
#give me the code to implement the grounding technique
start_time = time.time()
docs = retriever.retrieve(question)
retrieve_time = time.time() - start_time

start_time = time.time()
generation = rag_chain.invoke({"context": format_docs(docs), "question": question})
generation_time = time.time() - start_time

total_time = time.time() - start_time - retrieve_time

print(f"Retrieval Time: {retrieve_time:.4f} seconds")
print(f"Generation Time: {generation_time:.4f} seconds")
print(f"Total Processing Time: {retrieve_time + generation_time:.4f} seconds")
print("\nGeneration Result:")
print(generation)


def get_retriver():
    return retriever

def get_rag_chain():
    return rag_chain

def generate_embeddings_and_vector(texts):
    """
    Creates text embeddings using Vertex AI and builds a Chroma vector index

    Args:
        texts: A list of text chunks

    Returns:
        A Chroma vector index ready for similarity search
    """
    vector_index = Chroma.from_texts(texts, vertex_embeddings).as_retriever()
    return vector_index    


#=============hallucinator-grader and answer-grader======================
'''
# Hallucination Grader

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()
hallucination_grader.invoke({"documents": docs, "generation": generation})
print(hallucination_grader)

### Answer Grader

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()
answer_grader.invoke({"question": question, "generation": generation})
print(answer_grader)
'''


#=============router======================
'''
# Router

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import time
# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on an organization called beacon,its functions,its
    programs,its clinical research reports.Also, use vectorstore for answering questions on an ongoing project given by the beacon to a
    company called Cambridge Technology.This project consists of creating an application for helping parents who have children suffering from
    autism and other disabilities.Also, this project aims at proividing resources(preferably free) to parents about free food, free diapers, free legal aid etc.
    The information about the whole project namely the emails, the requirements, the tech stack used  , the deadlines, the codebases is present
    in the vectorstore.So, use vectorstore for any user question related to the project.
    Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
start_time = time.time()
question_router = prompt | llm | JsonOutputParser()
question = "where can i get free food in virginia"
docs = retriever.retrieve(question)
doc_txt = docs[1].node.text
print(question_router.invoke({"question": question}))
generation_time = time.time() - start_time
print(f'Generation Time: {generation_time:.4f} seconds')

'''

