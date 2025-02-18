import streamlit as st
import pinecone
from transformers import pipeline
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")


pinecone.init(api_key=pinecone_api_key, environment='us-west1-gcp')
index_name = 'tender-bot'


if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)  


model = pipeline('feature-extraction', model='bert-base-uncased')


def get_embeddings(texts):
    embeddings = []
    for text in texts:
        embedding = model(text)[0][0]  # Extracting the embedding
        embeddings.append(np.array(embedding))
    return embeddings


def insert_tenders_to_pinecone(tenders_df):
    tender_embeddings = get_embeddings(tenders_df['tender_text'].tolist())
    pinecone_index = pinecone.Index(index_name)
    
    for i, embedding in enumerate(tender_embeddings):
        pinecone_index.upsert([(str(i), embedding.tolist(), {'tender_id': str(i), 'text': tenders_df.iloc[i]['tender_text']})])


def retrieve_similar_tenders(query_text, top_k=5):
    query_embedding = get_embeddings([query_text])[0]
    pinecone_index = pinecone.Index(index_name)
    
    results = pinecone_index.query([query_embedding.tolist()], top_k=top_k, include_metadata=True)
    
    similar_tenders = []
    for match in results['matches']:
        tender = match['metadata']
        similar_tenders.append(tender['text'])
    
    return similar_tenders

st.title(" AI Tender Bot")

uploaded_file = st.file_uploader("Upload Old Tenders", type=["csv"])

if uploaded_file is not None:
    # Load the tenders
    tenders_df = pd.read_csv(uploaded_file)
    
    if 'tender_text' not in tenders_df.columns:
        st.error("CSV should contain a column named 'tender_text'.")
    else:
        st.write(f"Loaded {len(tenders_df)} tenders.")
        insert_tenders_to_pinecone(tenders_df)
        st.success("Tenders inserted into Pinecone.")

query = st.text_input("Enter a tender topic or description to generate similar tenders:")

if query:
    similar_tenders = retrieve_similar_tenders(query)
    
    if similar_tenders:
        st.write("Similar tenders found:")
        for tender in similar_tenders:
            st.write(f"- {tender}")
    else:
        st.write("No similar tenders found.")
