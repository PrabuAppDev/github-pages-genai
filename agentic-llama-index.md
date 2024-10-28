---
layout: default
title: vector-db
---

# Agentic (Llama Index)

<a href="https://github.com/PrabuAppDev/llama-index/blob/main/llama-index-agent.md" target="_blank">View the latest Agentic cookbook (Llama Index) on GitHub</a>

```python
!pip install openai
```

```python
import nest_asyncio
import os

# Apply nest_asyncio to handle nested event loops (useful for Jupyter notebooks)
nest_asyncio.apply()

# Ensure the OpenAI API key is set as an environment variable
assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

# Import the OpenAI and embedding classes from Llama-Index
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Initialize the language model (LLM) using gpt-4o-mini and embedding model
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
embed_model = OpenAIEmbedding()

# Set the LLM and embedding model globally for usage
Settings.llm = llm
Settings.embed_model = embed_model
```


```python
from llama_index.core import SimpleDirectoryReader

# Load the Acura MDX manual
acura_docs = SimpleDirectoryReader(input_files=["acura_mdx_manual.pdf"]).load_data()
```


```python
from llama_index.core import VectorStoreIndex

# Create vector store index from the Acura MDX manual
acura_index = VectorStoreIndex.from_documents(acura_docs)

# Create a query engine for the Acura manual
acura_query_engine = acura_index.as_query_engine(similarity_top_k=3)
```


```python
from IPython.display import display, HTML

# Example query: Ask about resetting the oil change light
response = acura_query_engine.query("How do I disable the automatic high beam feature on a 2022 Acura MDX?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```


<p style="font-size:20px">To disable the automatic high beam feature on a 2022 Acura MDX, set the power mode to ON while the vehicle is stationary. With the light switch in the AUTO position, pull the lever toward you and hold it for at least 40 seconds. Once the auto high-beam indicator light blinks twice, you can release the lever.</p>



```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker

# Define the query engine tool for Acura manual
query_engine_tools = [
    QueryEngineTool(
        query_engine=acura_query_engine,
        metadata=ToolMetadata(
            name="acura_manual",
            description="Provides information from the Acura MDX 2022 owner's manual",
        ),
    )
]

# Create a function-calling agent worker
agent_worker = FunctionCallingAgentWorker.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)

# Convert the agent worker to an agent
agent = agent_worker.as_agent()

# Use the agent to ask a question about the Acura manual
response = agent.chat("How do I disable the automatic high beam feature on a 2022 Acura MDX?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

    Added user message to memory: How do I disable the automatic high beam feature on a 2022 Acura MDX?
    === Calling Function ===
    Calling function: acura_manual with args: {"input": "disable automatic high beam feature 2022 Acura MDX"}
    === Function Output ===
    To disable the automatic high-beam feature on a 2022 Acura MDX, set the power mode to ON while the vehicle is stationary. With the light switch in the AUTO position, pull the lever toward you and hold it for at least 40 seconds. After the auto high-beam indicator light blinks twice, you can release the lever. This will turn off the auto high-beam system.
    === LLM Response ===
    To disable the automatic high-beam feature on a 2022 Acura MDX, follow these steps:
    
    1. Set the power mode to ON while the vehicle is stationary.
    2. With the light switch in the AUTO position, pull the lever toward you and hold it for at least 40 seconds.
    3. After the auto high-beam indicator light blinks twice, you can release the lever.
    
    This will turn off the auto high-beam system.
    


<p style="font-size:20px">To disable the automatic high-beam feature on a 2022 Acura MDX, follow these steps:

1. Set the power mode to ON while the vehicle is stationary.
2. With the light switch in the AUTO position, pull the lever toward you and hold it for at least 40 seconds.
3. After the auto high-beam indicator light blinks twice, you can release the lever.

This will turn off the auto high-beam system.</p>


# Agentic Architecture Overview

The setup using `FunctionCallingAgentWorker` with Llama Index can be considered an example of agentic architecture. Here's a breakdown of why this approach qualifies:

1. **Agents and Autonomy**  
   - The `FunctionCallingAgentWorker` creates an agent that autonomously decides which tools (query engines) to utilize based on the user's query.
   - This makes it an "agent" because it can perform actions independently to resolve queries. For instance, the agent autonomously decides which section of the manual to query to answer a question.

2. **Tool Integration**  
   - The agent is integrated with "tools" (`QueryEngineTool`), which provide specific capabilities—in this case, querying the Acura manual data.
   - This tool integration is central to agentic architecture as it allows the agent to perform specialized tasks using pre-defined functionalities.

3. **Reasoning and Function Calling**  
   - The `FunctionCallingAgentWorker` allows the agent to reason and call specific functions as needed based on the user's prompt.
   - This setup enables the agent to make decisions and take actions, such as querying the Acura manual for specific information like resetting the oil change light.

## Differences from a Basic Query System
- A basic query system only returns search results without processing or reasoning, while the agentic approach "thinks through" the required steps.
- Agentic architecture enables multiple decision-making steps and tool usage, adding sophistication beyond simple query-response mechanisms.

## Benefits
- **Modularity**: Additional tools can be added to the agent, enabling it to autonomously decide when to use each one.
- **Scalability**: The agent can scale to handle complex, multi-step queries and interactions, making it more versatile than a basic query engine.

In summary, this setup leverages principles of agentic architecture, enabling it to dynamically and autonomously interact with users' queries. This is beneficial for scenarios that require more than simple responses, making it capable of sophisticated, contextualized interactions.