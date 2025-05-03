from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("Ghosts.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Ghosts"] + " " + row["Their_manner_of_speaking"] + " " + row["What_each_ghost_knows_about_the_Thief"]+ " " + row["Each_ghosts_past_life"]+ " " + row["Each_ghosts_opinion_about_the_other_ghosts"],
            metadata={"Ghosts": row["Ghosts"], "Their_manner_of_speaking": row["Their_manner_of_speaking"], "What_each_ghost_knows_about_the_Thief": row["What_each_ghost_knows_about_the_Thief"], "Each_ghosts_past_life": row["Each_ghosts_past_life"],  "Each_ghosts_opinion_about_the_other_ghosts": row["Each_ghosts_opinion_about_the_other_ghosts"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="ghostthoughts",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)