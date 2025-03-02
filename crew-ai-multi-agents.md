---
layout: default  
title: crew-ai-multi-agents
---

# Crew AI-Powered Expense Manager

This document provides an overview of the LLM-based approach for the Crew AI-powered Expense Manager, with a comparison to the Regex-based approach. The focus is on leveraging Large Language Models (LLMs) for extracting, categorizing, and managing expenses from receipts using Crew AI.

## Available Documentation

- <a href="https://github.com/PrabuAppDev/agentic-crew-ai/blob/main/md/ocr-regex/crew-openai-receipt-organizer-ocr-regex.md" target="_blank">Regex-Based Crew AI Approach</a>
- <a href="https://github.com/PrabuAppDev/agentic-crew-ai/blob/main/md/ocr-crewai-llm/crew-openai-receipt-organizer-llm.md" target="_blank">LLM-Based Crew AI Approach</a>

## GitHub Repository

<a href="https://github.com/PrabuAppDev/agentic-crew-ai" target="_blank">Crew AI Expense Manager GitHub Repository</a>


## LLM-Based Crew AI Approach (Primary Focus)

### **Workflow Overview**

The Crew AI-powered Expense Manager follows a **task execution flow**:

1. Receipt Image Processing (EasyOCR) → Extracts raw text from receipt images.
2. Expense Extraction (LLM) → Parses the receipt text and extracts structured details (date, vendor, amount, items).
3. Expense Categorization (LLM) → Assigns a category to each transaction (e.g., Groceries, Gas, Dining).
4. Financial Record Keeping (LLM) → Stores structured expenses into a CSV file and detects potential fraud.

### ** Crew AI Agents & Their Roles**

#### Expense Extractor (LLM-Powered)

- Role: Extracts structured data from raw receipt text.
- Goal: Identify and extract date, vendor, amount, and items from receipts.
- Backstory: A specialized AI trained in financial document processing.

#### Expense Categorizer (LLM-Powered)

- Role: Assigns appropriate expense categories to transactions.
- Goal: Analyze extracted expense data and classify transactions (e.g., Food, Travel, Shopping).
- Backstory: A financial assistant AI with deep knowledge of expense classification.

#### Financial Record Keeper (LLM-Powered)

- Role: Stores structured expenses and detects fraudulent activity.
- Goal: Maintain a financial record of categorized expenses and flag suspicious transactions.
- Backstory: A security-focused AI capable of detecting anomalies in expense patterns.

### ** Crew AI Task & Execution Flow**

Below is a structured representation of how the Crew AI agents execute tasks:

1. Expense Extractor
   -  Starts with OCR-extracted text
   -  Extracts structured expense details (Date, Amount, Vendor, Items)
   -  Passes structured data to the Categorizer

2. Expense Categorizer
   -  Receives structured expense data
   -  Determines the correct category for each expense
   -  Passes categorized data to the Record Keeper

3. Financial Record Keeper
   -  Receives categorized expense data
   -  Stores structured data into CSV
   -  Detects anomalies for fraud analysis

## **Comparison: LLM-Based vs. Regex-Based Approach**

| Feature             | LLM-Based Approach (Preferred)                  | Regex-Based Approach                 |
| ------------------- | ----------------------------------------------- | ------------------------------------ |
| Flexibility     | ✅ Adapts to different receipt formats           | ❌ Limited to predefined patterns     |
| Fraud Detection | ✅ LLM analyzes transactions for anomalies       | ❌ Regex cannot detect fraud patterns |
| Maintenance     | ✅ Self-improving with better models             | ❌ Needs manual adjustments           |

## ** Next Steps**

- ✅ Complete testing of the LLM-powered Crew AI pipeline.
- Optimize task prompts for better accuracy.
- Implement fraud detection enhancements using anomaly detection models.
