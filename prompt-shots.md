---
layout: default
title: prompt-shots
---

# Prompt engineering with Zero-Shot Learning,  One-Shot, Few-Shot, Few-Shot with Instruction, Chain-of-Thought Prompting & Chain-of-Thought with Few-Shot Learning

<a href="https://github.com/PrabuAppDev/genai-rag/blob/main/prompt_engineering_shots.md" target="_blank">View the Prompt engineering (Open AI LLM API) on GitHub</a>

```python
!pip install openai
```

```python
# import os
# from openai import OpenAI
# https://platform.openai.com/docs/api-reference/debugging-requests?lang=python

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-4o-mini",
# )
```


```python
import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_response(messages, model="gpt-4o-mini", max_tokens=100, temperature=0.7):
    """
    General function to generate a response from OpenAI using the messages format.
    Uses gpt-4o-mini as the default model.
    
    Parameters:
    - messages: List of messages (chat format) where each message is a dict with "role" and "content"
    - model: The model to be used (default is gpt-4o-mini)
    - max_tokens: Maximum number of tokens to generate
    - temperature: Sampling temperature (creativity level)
    
    Returns:
    - Generated response as a string
    """
    response = client.chat.completions.create(
        model=model,  # Default to gpt-4o-mini
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Access the message content correctly using dot notation
    return response.choices[0].message.content.strip()

# Example usage of the generate function with different prompt patterns.
```


```python
# Zero-Shot Learning
# This method provides no examples and asks the model to generate a response based on its prior knowledge.
messages = [{"role": "user", "content": "Translate the word 'cat' into French."}]
response = generate_response(messages)
print(f"Zero-Shot Response: {response}")
```

    Zero-Shot Response: The word 'cat' in French is 'chat'.
    


```python
# One-Shot Learning
# This method provides one example and asks the model to use that example to guide its answer.
messages = [
    {"role": "user", "content": "Example: cat -> chat."},
    {"role": "user", "content": "Translate the word 'dog' into French."}
]
response = generate_response(messages)
print(f"One-Shot Response: {response}")
```

    One-Shot Response: The word 'dog' in French is 'chien'.
    


```python
# Few-Shot Learning
# This method provides multiple examples (usually 2-5) and asks the model to apply the pattern seen in the examples.
messages = [
    {"role": "user", "content": "Examples: cat -> chat, dog -> chien."},
    {"role": "user", "content": "Translate the word 'bird' into French."}
]
response = generate_response(messages)
print(f"Few-Shot Response: {response}")
```

    Few-Shot Response: The word 'bird' in French is 'oiseau'.
    


```python
# Few-Shot with Instructional Prompt
# This method provides examples along with explicit instructions on how the model should behave.
messages = [
    {"role": "user", "content": "Please translate the following words from English to French.\nExamples: cat -> chat, dog -> chien."},
    {"role": "user", "content": "Translate the word 'house' into French."}
]
response = generate_response(messages)
print(f"Few-Shot with Instruction Response: {response}")
```

    Few-Shot with Instruction Response: House in French is "maison."
    


```python
# Chain-of-Thought Prompting
# This method encourages the model to work through a problem step-by-step, mimicking how humans reason through tasks.
messages = [
    {"role": "user", "content": "First, multiply 10 by 14 to get 140. Then, multiply 2 by 14 to get 28. Now, add 140 and 28."},
    {"role": "user", "content": "What is 12 times 14?"}
]
response = generate_response(messages)
print(f"Chain-of-Thought Response: {response}")
```

    Chain-of-Thought Response: 12 times 14 is 168.
    


```python
# Chain-of-Thought with Few-Shot Learning
# This method combines examples with step-by-step reasoning to help the model understand and process more complex tasks.
messages = [
    {"role": "user", "content": "Example: 12 times 14: First, multiply 10 by 14 to get 140. Then, multiply 2 by 14 to get 28. Now, add 140 and 28 to get 168."},
    {"role": "user", "content": "Now, solve 13 times 15."}
]
response = generate_response(messages)
print(f"Chain-of-Thought with Few-Shot Response: {response}")
```

    Chain-of-Thought with Few-Shot Response: To solve 13 times 15, we can break it down as follows:
    
    First, multiply 10 by 15 to get 150.  
    Then, multiply 3 by 15 to get 45.  
    
    Now, add 150 and 45 to get:
    
    150 + 45 = 195.
    
    So, 13 times 15 equals 195.
    


```python
# Instruction-Tuning and Custom Prompting
# This method gives explicit instructions without examples, to guide the model's response based solely on those instructions.
messages = [{"role": "user", "content": "Summarize the following text in one sentence: 'The Eiffel Tower is located in Paris and was built in 1889.'"}]
response = generate_response(messages)
print(f"Instruction-Tuned Prompt Response: {response}")
```

    Instruction-Tuned Prompt Response: The Eiffel Tower, constructed in 1889, is situated in Paris.
    