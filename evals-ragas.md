---
layout: default  
title: crew-ai-multi-agents
---

### Grounding Analysis of Model Response using Retrieval-Augmented Generation Assessment (RAGAs)

**Grounding** refers to the model's ability to produce outputs that are:
1. **Factually accurate**: The response should reflect verified knowledge or facts, not fabricated information or hallucinations.
2. **Aligned with retrieved context**: The model must base its output on reliable sources, such as external data or retrieval systems, and ensure its response is grounded in actual knowledge, rather than speculative or invented details.


## 🔗 GitHub Repository

<a href="https://github.com/PrabuAppDev/evals-frameworks/blob/main/evals-ragas-notebook.ipynb" target="_blank">Ragas for evaluation</a>

![Ragas metrics](/assets/images/ragas-metrics.gif)

#### Breakdown of Grounding:
- **Factual Correctness**: The response accurately reflected known facts about spinach and chicken protein content, which is a core aspect of grounding.
- **Faithfulness**: While the model did a good job of staying faithful to the context (i.e., the instruction to avoid fabricated citations), there was some room for improvement. Faithfulness in grounding means ensuring that the model doesn't stray from the provided or retrieved context, and a score of 0.75 suggests the response is mostly grounded but not fully aligned with all retrieved context.
- **AspectCritic**: The fact that this result passed suggests that the response adhered to the general expectations and standards, indicating proper grounding.



### Evaluation Results Analysis

**Factual Correctness Result: 1.0**
- The model's response was **factually correct** and aligned well with the expected answer.
- The output correctly identified that there is no peer-reviewed study supporting the claim that spinach has more protein than chicken.
- The response provided relevant facts based on known data (e.g., USDA).

**Faithfulness Result: 0.75**
- The faithfulness score of **0.75** suggests that the model's response was **mostly faithful** to the retrieved context and the expected information.
- However, there is room for improvement to ensure the response is fully aligned with the given context, especially if the retrieved context could be more clearly reflected in the response.

**AspectCritic Result: 1**
- The **aspect critic** check passed, indicating that the response was **consistent** with the general criteria for a well-formed, factually supported answer.

### Summary:
- The model performed well in providing a **factually correct** response but can be further improved in terms of **faithfulness**. Ensuring that the retrieved context is fully reflected in the response would enhance faithfulness.
  
- The **aspect critic** check passed, confirming that the response adhered to the expected format.

This indicates that the approach is largely successful, with **hallucinations** avoided, but there is still room for improvement in **faithfulness**.

### Conclusion:
In short, the model's response was grounded in factual accuracy and retrieved information. The fact that it avoided hallucinations and provided an accurate factual comparison shows strong grounding. However, the slight dip in **faithfulness** suggests the response could have more closely matched the retrieved context. 

Thus, while the response is grounded, there’s still room for improvement in ensuring the response fully aligns with the data retrieved and context provided, especially in the case of more complex or multi-step queries.
