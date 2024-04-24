import chromadb
import os
import uuid
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import  PyPDFLoader
from langchain_core.documents import Document

x = os.getcwd()
print(x)

doc = r"""From Wikipedia, the free encyclopedia
For other uses, see Elon Musk (disambiguation).
Elon Musk
FRS
2018 cannabis incident
In September 2018, Musk was interviewed on an episode of The Joe Rogan Experience podcast, during which he sampled a cigar laced with cannabis.[303] In 2022, Musk said that he and other SpaceX employees had subsequently been required to undergo random drug tests for about a year following the incident, as required by the Drug-Free Workplace Act of 1988 for Federal contractors.[304] In a 2019 60 Minutes interview, Musk had said, "I do not smoke pot. As anybody who watched that podcast could tell, I have no idea how to smoke pot."[305]

Musicviduals have had more influence than Musk on life on Earth, and potentially life off Earth too".[560][561] In February 2022, Musk was elected to the National Academy of Engineering.[562] Following a tumultuous year of changes and controversies at X, The New Republic labeled Musk its 2023 Scoundrel of the Year.[563]")

"""

#loader = PyPDFLoader(x + '\sample1.pdf')


spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)

split_docs = spliter.split_text(doc)

    
openai_fn = embedding_functions.OpenAIEmbeddingFunction(api_key= os.getenv('OPENAI_API_KEY'), model_name="text-embedding-ada-002")
    
client = chromadb.HttpClient(host='localhost', port=8000)
print(client.list_collections())

#collection2 =  client.get_or_create_collection(name="ElonMusk", embedding_function=openai_fn)
collection2 = client.get_collection(name="ElonMusk", embedding_function=openai_fn)

for doc in split_docs:
    collection2.add(
        ids=[str(uuid.uuid1())], metadatas=[{"source": str(uuid.uuid1())}], documents=doc
    )

results = collection2.query(
    query_texts=["The family was wealthy during Elon's youth"],
    n_results=2
)

print(results)