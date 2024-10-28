---
layout: default
title: vector-db
---

# Vector DB cookbook (Pinecone)

<a href="https://github.com/PrabuAppDev/genai-vector-index/blob/main/vector-pinecone-cookbook-v2.md" target="_blank">View the latest Vector DB (Open AI LLM API & Pinecone) on GitHub</a>

```python
# Required Imports
import os
import openai
from pinecone import Pinecone, ServerlessSpec
```


```python
from pinecone import Pinecone, ServerlessSpec
import os

# Initialize Pinecone instance using the updated method
try:
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")  # Fetching from environment variables
    )

    # Create or connect to the index
    index_name = "satellite-search"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # Replace with the actual dimension of your embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",   # Use a cloud provider supported by your plan (e.g., 'aws', 'gcp')
                region="us-east-1"  # Update to a region available on your plan (check Pinecone Console for options)
            )
        )

    # Connect to the index
    index = pc.Index(index_name)
    print("Pinecone Index connected successfully!")
except Exception as e:
    print("Pinecone Initialization Failed:", e)
```

    Pinecone Index connected successfully!
    


```python
from openai import OpenAI

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to embed a document using OpenAI Client
def embed_document(text):
    token_count = count_tokens(text)
    if token_count < 1000:
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=[text]
            )
            # Correct way to access the embeddings
            embeddings = response.data[0].embedding
            print("Embedding successful:", embeddings[:5])  # Show a sample of the embedding
            return embeddings
        except Exception as e:
            print("OpenAI request failed:", e)
            return None
    else:
        print("Input too lengthy, please shorten it.")
        return None

# Example usage
text_input = "New developments in AI enhance satellite technology."
embedding = embed_document(text_input)

# Check if embedding was successful
if isinstance(embedding, list):
    store_embedding_in_pinecone("doc1", embedding)
else:
    print("Error: Embedding is not in a valid list format.")
```

    Token count: 8
    Embedding successful: [-0.002322586951777339, 0.007827353663742542, 0.00771274184808135, -0.008097030222415924, 0.0038226612377911806]
    Embedding for 'doc1' stored successfully!
    


```python
# Function to search Pinecone using a query text
def search_in_pinecone(query_text, top_k=3):
    query_embedding = embed_document(query_text)
    
    if not isinstance(query_embedding, list):
        print("Error generating query embedding.")
        return
    
    # Perform the query
    try:
        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace="ns1",
            include_metadata=True
        )
        
        # Display search results
        print("\nSearch Results:")
        if result and result.get('matches'):
            for match in result['matches']:
                print(f"ID: {match['id']}, Score: {match['score']}, Text: {match['metadata'].get('text', 'N/A')}")
        else:
            print("No results found.")
    except Exception as e:
        print("Failed to search in Pinecone:", e)

# Example usage
search_in_pinecone("How AI enhances satellite technology.")
```

    Token count: 6
    Embedding successful: [0.0040581729263067245, 0.004708980210125446, 0.000188044024980627, -0.006528513506054878, -0.0011840597726404667]
    
    Search Results:
    ID: doc1, Score: 0.973274291, Text: New developments in AI enhance satellite technology.
    ID: doc2, Score: 0.944378734, Text: New developments in AI enhance satellite technology.
    ID: doc3, Score: 0.931232691, Text: New developments in AI enhance satellite technology.
    


```python
# Function to search without metadata filters for troubleshooting
def enhanced_vector_search_no_filter(query_text, top_k=3):
    # Embed the query text
    query_embedding = embed_document(query_text)
    if not isinstance(query_embedding, list):
        print("Error generating query embedding.")
        return
    
    # Perform the search without any metadata filters
    try:
        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace="ns1",
            include_metadata=True
        )
        print("\n=== Search Results ===")
        if result.get("matches"):
            for match in result["matches"]:
                doc_id = match["id"]
                score = match["score"]
                text = match["metadata"].get("text")
                print(f"ID: {doc_id}, Score: {score}, Text: {text}")
        else:
            print("No matches found.")
    except Exception as e:
        print("Error during vector search:", e)

# Example usage
enhanced_vector_search_no_filter("Latest advancements in satellite technology.", top_k=3)
```

    Token count: 6
    Embedding successful: [-0.006524303928017616, 0.013479744084179401, 0.008299373090267181, -0.007760452572256327, -0.002101789228618145]
    
    === Search Results ===
    ID: doc1, Score: 0.93795687, Text: New developments in AI enhance satellite technology.
    ID: doc3, Score: 0.91320616, Text: New developments in AI enhance satellite technology.
    ID: doc2, Score: 0.89876163, Text: New developments in AI enhance satellite technology.
    