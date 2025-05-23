from pprint import pprint
from typing import List, Annotated

#importing time for analayzing speed of response
import time

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


from langchain_groq import ChatGroq




os.environ["GROQ_API_KEY"] = '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'
os.environ["TAVILY_API_KEY"] = "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"

# Search
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)


# LLM
local_llm = "llama3:latest"
local_llm= "llama3.2:1b-instruct-q4_K_M"

# LLM
#llm = ChatOllama(model=local_llm, format="json", temperature=0)


# Set environment variables
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC-ASsI6zwI9UiDcR9xqEH7SyeHl2MS8HY'

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
 
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


# Retrieval Grader


llm = ChatOllama(model=local_llm, temperature=0)
'''llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,)'''

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




#=============Generator======================


#Generator
# Generate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use required number of sentences to keep the answer well detailed <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

#llm = ChatOllama(model=local_llm, temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.node.text for doc in docs) 


# Chain
rag_chain = prompt | llm | StrOutputParser()





#=============hallucinator-grader and answer-grader======================

# Hallucination Grader

# LLM
#llm = ChatOllama(model=local_llm, format="json", temperature=0)

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


### Answer Grader

# LLM
#llm = ChatOllama(model=local_llm, format="json", temperature=0)

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




#=============router======================

# Router

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import time
# LLM
#llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Router with corrected prompt template
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on an organization called beacon,its functions,its
    programs,its clinical research reports.Also, use vectorstore for answering questions on an ongoing project given by the beacon to a
    company called Cambridge Technology.This project consists of creating an application for helping parents who have children suffering from
    autism and other disabilities.Also, this project aims at proividing resources(preferably free) to parents about free food, free diapers, free legal aid etc.
    The information about the whole project namely the emails, the requirements, the tech stack used, the deadlines, the codebases is present
    in the vectorstore.So, use vectorstore for any user question related to the project.
    Otherwise, use web-search. 
    
    You must respond with a JSON object that has a single key 'datasource' with only one of two possible values:
    - 'vectorstore' for questions about Beacon or the Cambridge Technology project
    - 'web_search' for all other questions
    
    Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()






# State


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    #from langgraph.graph.message import add_messages
    documents: List[str]
    #documents: Annotated[list, add_messages]


#---------------------------Nodes--------------------------------


# Modify the retrieve function to convert NodeWithScore to Document
def retrieve(state):
    """
    Retrieve documents from vectorstore and convert to compatible format

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    nodes = retriever.retrieve(question)
    
    # Convert NodeWithScore objects to Documents
    documents = [
        Document(page_content=node.node.text) for node in nodes
    ]
    
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Extract text content from documents
    docs_content = "\n\n".join(doc.page_content for doc in documents)

    # RAG generation
    generation = rag_chain.invoke({"context": docs_content, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


# Modify the grade_documents function to work with Documents
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]  #documents = state.get("documents", []) 

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


# Conditional edge


# Update the route_question function with better error handling
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(f"Routing question: {question}")
    
    try:
        source = question_router.invoke({"question": question})
        print(f"Router response: {source}")
        
        # Validate the response format
        if not isinstance(source, dict) or "datasource" not in source:
            print("Invalid router response format, defaulting to web search")
            return "websearch"
            
        datasource = source["datasource"].lower()
        if datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        else:
            print(f"Unknown datasource '{datasource}', defaulting to web search")
            return "websearch"
            
    except Exception as e:
        print(f"Error in routing: {e}")
        return "websearch"  # Default to web search on error


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


# Conditional edge(conditional edges are nothing but functions which will take a decision/route towards a node at a given point of time)


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    try:
        # Extract text from documents (now all are Document objects)
        documents_text = "\n\n".join(doc.page_content for doc in documents)
        
        print("Documents Text:", documents_text[:500] + "..." if len(documents_text) > 500 else documents_text)
        
        score = hallucination_grader.invoke(
            {"documents": documents_text, "generation": generation}
        )
        
        print("Hallucination Grader Score:", score)
        grade = score.get('score', 'no')

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            
            try:
                answer_score = answer_grader.invoke({"question": question, "generation": generation})
                print("Answer Grader Output:", answer_score)
                answer_grade = answer_score.get('score', 'no')
                
                if answer_grade == "yes":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            
            except Exception as e:
                print(f"Error in answer grading: {e}")
                return "not useful"
        
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    
    except Exception as e:
        print(f"Error in hallucination grading: {e}")
        return "not supported"

workflow = StateGraph(GraphState)

# Define the nodes(nodes are nothing but functions which will achieve a certain goal)
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae




#----------------------build graph-------------------------------


# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)


#-------------------------COMPILATION-------------------------------


from pprint import pprint

# Compile
app = workflow.compile()
inputs = {"question": "What is the tech stack used in the project, explain in detail"}
#Explain about the grounding technique used to enhance the beacon chatbot
#
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])



#groqCLoud api key = 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
