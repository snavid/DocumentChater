import chromadb
import os
import uuid
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import  PyPDFLoader
from langchain_core.documents import Document

x = os.getcwd()
print(x)


loader = PyPDFLoader(x + '\websites\sample1.pdf')

spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)
doc = loader.load()
split_docs = spliter.split_documents(doc)

print(split_docs[0])
openai_fn = embedding_functions.OpenAIEmbeddingFunction(api_key= os.getenv('OPENAI_API_KEY'), model_name="text-embedding-ada-002")
    
client = chromadb.HttpClient(host='localhost', port=8000)
print(client.list_collections())

collection2 =  client.get_or_create_collection(name="test", embedding_function=openai_fn)
#collection2 = client.get_collection(name="ElonMusk", embedding_function=openai_fn)

for doc in split_docs:
    collection2.add(
        ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
    )

results = collection2.query(
    query_texts=["what is risk management"],
    n_results=2
)

print(results)
