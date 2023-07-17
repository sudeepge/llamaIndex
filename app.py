import pinecone
import os, sys
import gradio as gr  
import logging
import openai
#import umap
import matplotlib.pyplot as plt
import cProfile
from dotenv import load_dotenv
#from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from llama_index import GPTVectorStoreIndex, GPTListIndex, SimpleDirectoryReader, ServiceContext, LLMPredictor, PromptHelper
from llama_index import VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import Anthropic
from llama_index.llms import OpenAI
from nomic import atlas
import nomic
import numpy as np



# # Query function
# def query_index(query):

#   # Profile index operations
#   cProfile.run('query_tokens = llama_model.tokenize_docs([query])')
#   cProfile.run('results = index.query(query_tokens[0].embeddings, top_k=5)')

#   # Log results
#   logging.debug(f'Query tokens: {query_tokens}')
#   logging.debug(f'Top 5 results: {results.ids[:5]}')

#   # Visualize projections
#   #projected = umap.UMAP().fit_transform(doc_embeddings)
#   #plt.scatter(projected[:,0], projected[:,1])
#   #plt.savefig('projection.png')

#   return results


query_examples=[
    ("What is iMerit's biggest strength?"),
    ("How did CrowdReason benefit by working with iMerit?"),
    ("What do you mean by human-in-the-loop workflows?")
]

model_name1="gpt-3.5-turbo"
model_name2="claude-instant-1"
pinecone_vector_index='doc-index'
env_set=0
index1=0
index2=0

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def construct_index(directory_path):
    openai_max_input_size = 4096
    openai_num_outputs = 512
    openai_max_chunk_overlap = 0.9
    openai_chunk_size_limit = 600

    anthropic_max_input_size = 100000
    anthropic_num_output = 2048
    anthropic_max_chunk_overlap = 0.9

    load_dotenv()

    prompt_helper = PromptHelper(openai_max_input_size, openai_num_outputs, openai_max_chunk_overlap, chunk_size_limit=openai_chunk_size_limit)
    prompt_helper_anthropic = PromptHelper(anthropic_max_input_size, anthropic_num_output, anthropic_max_chunk_overlap)

    storage_context_index2=StorageContext.from_defaults()
    storage_context_index1 = StorageContext.from_defaults()


    llm_predictor1 = LLMPredictor(llm=OpenAI(temperature=1, model_name=model_name1, max_tokens=openai_num_outputs)) #gpt-3.5-turbo
    llm_predictor2 = LLMPredictor(llm = Anthropic(model=model_name2)) 
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor2, prompt_helper=prompt_helper_anthropic, chunk_size=95000)   

    # Pinecone setup
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    if ((pinecone.list_indexes()) != [pinecone_vector_index]):
        pinecone.create_index(pinecone_vector_index, dimension=1536, metric="euclidean", pod_type="p1")
    pinecone_index = pinecone.Index(pinecone_vector_index)

    documents = SimpleDirectoryReader(directory_path).load_data()
    index1= VectorStoreIndex(documents, llm_predictor=llm_predictor1, prompt_helper=prompt_helper, storage_context=storage_context_index1)
    index1.set_index_id = "index1"
    index1.storage_context.persist(persist_dir="./storage1")
    #index2 = VectorStoreIndex(documents, llm_predictor=llm_predictor2, prompt_helper=prompt_helper)

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context_index2 = StorageContext.from_defaults(vector_store=vector_store)
    index2 = VectorStoreIndex.from_documents(documents, storage_context=storage_context_index2)
    index2.storage_context.persist(persist_dir="./storage2")

      
    return index1



def get_ids_from_query(index,input_vector):
  print("searching pinecone...")
  results = index.query(vector=input_vector, top_k=10000,include_values=False)
  ids = set()
  print(type(results))
  for result in results['matches']:
    ids.add(result['id'])
  return ids

def get_all_ids_from_index(index, num_dimensions, namespace=""):
  num_vectors = index.describe_index_stats()["namespaces"][namespace]['vector_count']
  all_ids = set()
  while len(all_ids) < num_vectors:
    print("Length of ids list is shorter than the number of total vectors...")
    input_vector = np.random.rand(num_dimensions).tolist()
    print("creating random vector...")
    ids = get_ids_from_query(index,input_vector)
    print("getting ids from a vector query...")
    all_ids.update(ids)
    print("updating ids set...")
    print(f"Collected {len(all_ids)} ids out of {num_vectors}.")

  return all_ids

# all_ids = get_all_ids_from_index(index, num_dimensions=1536, namespace="")
# print(all_ids)

def visualize(index, num_embeddings):
    load_dotenv()
    nomic.login(os.environ.get('NOMIC_API_KEY'))

    print("Visualizing embedding count", num_embeddings)
    print(index.describe_index_stats())


    all_ids = get_all_ids_from_index(index, num_dimensions=1536, namespace="")
    id_list=list(all_ids)
    vectors = index.fetch(id_list)
    

    ids = []
    embeddings = []
    for id, vector in vectors['vectors'].items():
        ids.append(id)
        embeddings.append(vector['values'])

    embeddings = np.array(embeddings)


    atlas.map_embeddings(embeddings=embeddings, data=[{'id': id} for id in ids], id_field='id')



def chatbot_init():
    load_dotenv()
    storage_context_index1 = StorageContext.from_defaults(persist_dir="./storage1")
    index1 =  load_index_from_storage(storage_context_index1)

    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    pinecone_index = pinecone.Index('doc-index')
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context_index2 = StorageContext.from_defaults(vector_store=vector_store)
    index2 = VectorStoreIndex([], storage_context=storage_context_index2)


    if ((pinecone.list_indexes()) == [pinecone_vector_index]):
        print(pinecone.describe_index("doc-index"))
        print(pinecone_index.describe_index_stats())
        num_vectors = pinecone_index.describe_index_stats()["namespaces"][""]['vector_count']
        print(num_vectors)
        visualize(pinecone_index, num_vectors)

   
    #indices = [index1, index2]
    #playground = Playground(indices=indices)

    # response1 = index1.query(input_text, response_mode="default")
    # response2 = index2.query(input_text, response_mode="default")

    #playground.compare(input_text)
    return index1, index2


def chatbot(input_text):
    global env_set, index1, index2
    if env_set == 0:
        index1, index2 = chatbot_init()
        env_set=1


    query_engine1 = index1.as_query_engine()
    response1 = query_engine1.query(input_text)
    query_engine2 = index2.as_query_engine()
    response2 = query_engine2.query(input_text)
    
    return response1.response, response2.response



iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your query"),
                     outputs=[gr.components.Textbox(label=model_name1+ " Output"), gr.components.Textbox(label=model_name2+" Output")],
                     #outputs="text",
                     examples=query_examples,                         
                     title="Chat about iMerit's case studies", 
                     description="Trained on iMerit's publicly available case studies",
                    )

#index = construct_index("docs")
iface.launch(share=False)
