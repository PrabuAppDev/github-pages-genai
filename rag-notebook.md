---
layout: default
title: rag-notebook
---

# RAG Tutorials

<a href="https://github.com/PrabuAppDev/genai-rag/blob/main/rag-openai.md" target="_blank">View the latest RAG tutorial (Open AI LLM API) on GitHub</a>
Compare that for accuracy with a local LLM (scroll further down)

```python
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract text from a local PDF file."""
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Path to the local PDF file
pdf_path = 'acura_mdx_manual.pdf'

# Extract text from the PDF
manual_text = extract_text_from_pdf(pdf_path)
print("PDF text extracted.")
```

    PDF text extracted.
    


```python
import openai

def generate_embeddings(text):
    """Generate embeddings using OpenAI's latest API."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",  # The embedding model
        input=text
    )
    
    # Access the embeddings correctly using response.data
    embeddings = response.data[0].embedding
    return embeddings
```


```python
def search_relevant_content(query, manual_text):
    """Generate query embeddings and perform similarity search (mocking RAG)."""
    query_embedding = generate_embeddings(query)
    
    # Here you would perform the similarity search using FAISS or a similar tool
    # For now, we'll return a mocked section of the manual text
    return manual_text[:1000]  # Mock returning the first part of the text

# Example query:
query = "How do I reset the oil change light on a 2022 Acura MDX?"
relevant_content = search_relevant_content(query, manual_text)
print("Relevant content found:", relevant_content)
```

    Relevant content found: 2022 MDX Owner’s ManualEvent Data Recorders
    This vehicle is equipped with an event data recorder (EDR).  
    The main purpose of an EDR is to record, in certain crash or near 
    crash-like situations, such as an air bag deployment or hitting a road obstacle, data that will assist in understanding how a vehicle’s 
    systems performed. The EDR is designed to record data related 
    to vehicle dynamics and safety systems for a short period of 
    time, typically 30 seconds or less. The EDR in this vehicle is 
    designed to record such data as:
    •How various systems in your  vehicle were operating;
    •Whether or not the driver and passenger safety belts were 
    buckled/fastened;
    •How far (if at all) the driver was depressing the accelerator 
    and/or brake pedal; and,
    •How fast the vehicle was traveling.22 ACURA MDX-31TYA6011.book  0 ページ  ２０２１年１２月１４日　火曜日　午前１０時４２分These data can help provide a better understanding of the 
    circumstances in which crashes and injuries occur. NOTE: EDR data are recorded by your vehic
    


```python
import os
from openai import OpenAI

# Instantiate the client using environment variable for API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Check if the API key is loaded properly
if not client.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
```


```python
def query_openai_with_rag(query, relevant_content):
    """Query OpenAI API using RAG and gpt-4o-mini in chat format."""
    prompt = f"Based on the owner's manual, answer the following question:\n\nManual Section: {relevant_content}\n\nQuestion: {query}\n\nAnswer:"
    
    # Call the new chat completion API method
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Chat-based model
        messages=[{"role": "user", "content": prompt}],  # Pass the prompt in chat message format
        max_tokens=100,
        temperature=0.7
    )
    
    # Convert response to dictionary format if needed
    return response.model_dump()['choices'][0]['message']['content'].strip()

# Example query
query = "How do I reset the oil change light on a 2022 Acura MDX?"
relevant_content = "This is a section of the manual..."  # Mock content
response = query_openai_with_rag(query, relevant_content)
print(f"Query: {query}")
print(f"Response: {response}\n")

# Example query
query = "When does the Auto high beam mode turn ON?"
relevant_content = "This is a section of the manual..."  # Mock content
response = query_openai_with_rag(query, relevant_content)
print(f"Query: {query}")
print(f"Response: {response}")
```

    Query: How do I reset the oil change light on a 2022 Acura MDX?
    Response: To reset the oil change light on a 2022 Acura MDX, follow these steps:
    
    1. Turn on the ignition without starting the engine. This can be done by pressing the start button twice without pressing the brake pedal.
    2. Use the steering wheel controls to navigate to the "Settings" menu on the display.
    3. Select "Maintenance."
    4. Choose "Oil Life" from the options.
    5. Select "Reset" and confirm the action when prompted.
    
    This should reset the oil
    
    Query: When does the Auto high beam mode turn ON?
    Response: The Auto high beam mode typically turns ON when the vehicle is in low-light conditions, such as at night or in dark environments, and when no other vehicles are detected in front of you. It automatically switches from high beams to low beams when it detects the headlights of oncoming vehicles or taillights of vehicles in front of you. Always refer to the specific owner's manual for your vehicle for precise details and conditions.

<a href="https://github.com/PrabuAppDev/genai-rag/blob/main/rag-101.md" target="_blank">View the latest RAG tutorial (local LLM) on GitHub</a>

```python
!pip install transformers datasets faiss-cpu sentence-transformers
```

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import requests
import PyPDF2
```


```python
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load a smaller GPT-Neo model
model_name = "EleutherAI/gpt-neo-125M"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
```




    GPTNeoForCausalLM(
      (transformer): GPTNeoModel(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(2048, 768)
        (drop): Dropout(p=0.0, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPTNeoBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPTNeoAttention(
              (attention): GPTNeoSelfAttention(
                (attn_dropout): Dropout(p=0.0, inplace=False)
                (resid_dropout): Dropout(p=0.0, inplace=False)
                (k_proj): Linear(in_features=768, out_features=768, bias=False)
                (v_proj): Linear(in_features=768, out_features=768, bias=False)
                (q_proj): Linear(in_features=768, out_features=768, bias=False)
                (out_proj): Linear(in_features=768, out_features=768, bias=True)
              )
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPTNeoMLP(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )




```python
# Load a Sentence Transformer model to convert sentences to embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Dummy corpus of documents for retrieval
documents = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space."
    # "Artificial Intelligence is transforming the world.",
    # "The Grand Canyon is one of the seven wonders of the world."
]

# Convert the documents into embeddings
document_embeddings = embedder.encode(documents)

# Initialize FAISS index
dimension = document_embeddings.shape[1]  # Dimension of the embeddings
faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance index
faiss_index.add(np.array(document_embeddings))  # Add document embeddings to index

```


```python
def generate_without_rag(query):
    """Generate a response from the LLM without any external document retrieval."""
    # Tokenize the query and generate response
    inputs = tokenizer(query, return_tensors="pt").to(device)

    # Generate output without any retrieved document
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and return the generated response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def generate_with_rag(query):
    """Generate a response from the LLM with document retrieval using FAISS."""
    # Retrieve the relevant document using FAISS
    query_embedding = embedder.encode([query])
    _, indices = faiss_index.search(np.array(query_embedding), k=1)
    relevant_document = documents[indices[0][0]]
    
    # Augment the query with the retrieved document
    augmented_prompt = f"Using the information from the document, answer the question.\n\nDocument: {relevant_document}\n\nQuestion: {query}\n\nAnswer:"
    
    # Tokenize the augmented prompt and generate response
    inputs = tokenizer(augmented_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3  # This prevents repeating the same n-grams
    )

    # Decode and return the generated response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
query = "Where is the Eiffel Tower located?"

# Generate response without RAG
response_without_rag = generate_without_rag(query)
print("Response WITHOUT RAG:")
print(response_without_rag)

# Generate response with RAG
response_with_rag = generate_with_rag(query)
print("\nResponse WITH RAG:")
print(response_with_rag)

```

    Response WITHOUT RAG:
    Where is the Eiffel Tower located?
    
    The Eiffel Tower is located in the city of Berlin, Germany. The tower is located in the city of Berlin, Germany. The tower is a part of the city of Berlin, Germany
    
    Response WITH RAG:
    Using the information from the document, answer the question.
    
    Document: The Eiffel Tower is located in Paris.
    
    Question: Where is the Eiffel Tower located?
    
    Answer: The Tower is in Paris, France.
    


```python
def download_pdf(url, output_path='manual.pdf'):
    response = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(response.content)
    print(f"PDF downloaded and saved as {output_path}")

def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Download the PDF and extract text
url = "https://techinfo.honda.com/rjanisis/pubs/OM/AH/BTYA2222OM/enu/BTYA2222OM.pdf"
download_pdf(url, 'acura_mdx_manual.pdf')
manual_text = extract_text_from_pdf('acura_mdx_manual.pdf')
print("PDF text extracted.")

```

    PDF downloaded and saved as acura_mdx_manual.pdf
    PDF text extracted.
    


```python
def chunk_text(text, chunk_size=500):
    """Chunk the text into smaller pieces for embedding."""
    text_chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        text_chunks.append(chunk)
    return text_chunks

# Chunk the extracted text
text_chunks = chunk_text(manual_text)
print(f"Manual text chunked into {len(text_chunks)} chunks.")
```

    Manual text chunked into 273 chunks.
    


```python
# Load a pre-trained sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each text chunk
chunk_embeddings = embedder.encode(text_chunks)

# Store embeddings using FAISS for efficient similarity search
dimension = chunk_embeddings.shape[1]  # Dimension of the embeddings
faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance index
faiss_index.add(np.array(chunk_embeddings))

print("Embeddings created and indexed.")
```

    Embeddings created and indexed.
    


```python
def generate_response_with_llm(query, context):
    """Generate a response based on the relevant manual chunk using the LLM."""
    prompt = f"Using the information from the owner's manual section below, answer the question concisely:\n\nManual Section: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the input and set attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    attention_mask = inputs['attention_mask']
    
    # Set the `pad_token_id` to `eos_token_id`
    pad_token_id = tokenizer.eos_token_id
    
    # Generate the response with limited new tokens, attention mask, and no_repeat_ngram_size to prevent repetitions
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=attention_mask, 
        max_new_tokens=100,  # Control the number of new tokens generated
        pad_token_id=pad_token_id,
        no_repeat_ngram_size=3  # Prevent repetition of 3-grams
    )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
```


```python
def search_owners_manual(query, top_k=1):
    """Search for the most relevant chunk based on the query in the owner's manual."""
    query_embedding = embedder.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    results = [text_chunks[idx] for idx in indices[0]]
    return results

# Use the existing function `generate_response_with_llm` to generate responses

def query_acura_manual(query):
    """Query the Acura MDX owner's manual using embeddings and generate a response with the LLM."""
    # Search the owner's manual for relevant chunks
    result_chunks = search_owners_manual(query)
    context = result_chunks[0] if result_chunks else "No relevant information found."
    
    # Generate a response using the existing LLM function
    response = generate_response_with_llm(query, context)
    return response

# Example query
user_query = "When does the Auto high beam mode turn ON?"
response = query_acura_manual(user_query)
print(f"Final Response: {response}")
```

    Final Response: Using the information from the owner's manual section below, answer the question concisely:
    
    Manual Section: beam: All of the following conditions must be met before the high beams turn on. ●Your vehicle speed is 25mph (40 km/h) or more. ●There are no preceding or oncoming vehicle with headlights or taillights turned on. ●There are few street lights on the road ahead.One of the following conditions must be met before the low beams turn on. ●Your vehicle speed is 15 mph (24 km/h) or less. ●There is a preceding or oncoming vehicle with headlights or taillights turned on. ●There are many street lights on the road ahead.1How to Use the Auto High-Beam In the following cases, th e auto high-beam system may not switch the head lights properly or the switching timing may be ch anged. In case of the automatic switching operati on does not fit for your driving habits, please swit ch the headlights manually. •The brightness of the lights from the preceding or oncoming vehicle is intense or poor. •Visibility is poor due to the weather (rain, snow, fog, windshield frost, etc.). •Surrounding light sources, such as street lights, electric billboards and traf fic lights are illuminating the road ahead. •The brightness level of th e road ahead constantly changes. •The road is bumpy or has many curves. •A vehicle suddenly appears in front of you, or a vehicle in front of you is not in the preceding or oncoming direction. •Your vehicle is tilted with a heavy load in the rear. •A traffic sign, mirror, or other reflective object ahead is reflecting strong light toward the vehicle. •The oncoming vehicle freq uently disappears under roadside trees or be hind median barriers. •The preceding or oncoming vehicle is a motorcycle, bicycle, mobility scooter, or other small vehicle. The auto high-beam system keeps the headlight low beam when: •Windshield wipers are op erating at a high speed. •The camera has detected dense fog.22 ACURA MDX-31TYA6011.book 179 ページ ２０２１年１２月１４日 火曜日 午前１０時４２分uuOperating the Switches Around the Steering Wheel uAuto High-Beam 180Controls You can turn the auto high-beam system off. If you want to turn the system off or on, set the power mode to ON, then carry out the following procedures while the vehicle is stationary. To turn the system off:With the light switch is in AUTO, pull the lever toward you and hold it for at least 40 seconds. After the auto high-beam indicator light blinks twice, release the lever. To turn the system on: With the light switch is in AUTO, pull the lever toward you and hold it for at least 30 seconds. After the auto high-beam indicator light blinks once, release the lever.■How to Turn Off the Auto High-Beam1How to Use the Auto High-Beam If the Some driver assist systems cannot operate: Camera temperature too high message appears: •Use the climate control system to cool down the interior and, if necessary, also use de froster mode with the airflow directed toward the camera. •Start driving the vehicle to lower the windshield temperature, which cool s down the area around the
    
    Question: When does the Auto high beam mode turn ON?
    
    Answer: When the auto low-beam mode turns ON, the auto High-beam switch is turned OFF.
    
    How to turn off the Auto Low-Beams When the Auto low-beams are turned ON, you can turn off Auto High Beam.
    1How To Turn Off Auto High Beams When The Auto High beam is turned ON.
    2How To turn off auto High Beam When the High beam turns ON. The Auto high-beam switch is not turned OFF, but
    
