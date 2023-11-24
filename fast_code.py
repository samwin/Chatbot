from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.llms import LlamaCPP
from fastapi import FastAPI
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from fastapi.middleware.cors import CORSMiddleware
from llama_index.memory import ChatMemoryBuffer
import chromadb
import os
from pydantic import BaseModel
import re
os.environ['OPENAI_API_KEY'] = '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'


class Query(BaseModel):
    text : str


# llm  = LlamaCPP(
#     model_path='/root/.cache/gpt4all/mistral-7b-openorca.Q4_0.gguf',
#     temperature=0.5,
#     context_window=3900,
#     max_new_tokens=100, 
#     generate_kwargs={},
#     model_kwargs={"n_gpu_layers": 50},
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
# )

#action plan db till page 12
db1 = chromadb.PersistentClient(path='action_plan')
chroma_collection1 = db1.get_or_create_collection('action')
vector_store1 = ChromaVectorStore(chroma_collection=chroma_collection1)
index1 = VectorStoreIndex.from_vector_store(vector_store=vector_store1)
query_engine1 = index1.as_query_engine()


#full manual for course intro
db = chromadb.PersistentClient('efficient_nodes')
chroma_collection = db.get_or_create_collection('efficient')
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()

#function to page number of course manual
page_number_regex = r"'page_number':\s*'(\d+)'"
def get_page_numbers(response, public_url):
    source = []
    for node in response.source_nodes:
        match = re.search(page_number_regex, node.text)
        source.append(public_url+f"#page="+str(int(match.group(1)) + 1))
    return source


#functoin to get page number of action plan
def get_sources(response, public_url : str):
    sources = []
    for node in response.source_nodes:
        sources.append(f'{public_url}#page={node.metadata["page_label"]}')        
    return sources





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials = True,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.get('/')
def hello():
    return {"hello" : 'api'}


@app.post('/course')
def course_prod(text : Query):
    public_url = "https://storage.googleapis.com/egdk_manuals/Manual%20for%20course%20administration%20(1).pdf"   
    
    response = query_engine.query(f'{text.text}')
    source = get_page_numbers(response=response, public_url=public_url)
    print(response)
    return {'response' : f'{response}', 'source' : f'{source}'}



@app.post('/action')
def action_prod(text : Query):
    public_url = "https://storage.googleapis.com/egdk_manuals/Action%20plan%20module%20manual.pdf"
    response = query_engine1.query(f'{text.text}')
    source = get_sources(response=response, public_url=public_url)
    return {'response' : f'{response}', 'source' : f'{source}'}
    






