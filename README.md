# 📈 RAG for Nha Trang University Admission Score Queries
This project is designed to help users quickly retrieve and interact with admission scores of Nha Trang University for the years 2024 and 2025. It enables students to easily check annual admission benchmarks and make more informed decisions when selecting their preferred majors or programs.

## 📌 Objectives
- Use the **BAAI/bge-small-en-v1.5** embedding model to convert the 2024 and 2025 admission score data into a vector database.  
- Leverage the language understanding capabilities of the **Gemini** large language model, combined with a custom vector database, to provide fast and accurate information retrieval. 

---

## 🛠️ Technologies Used
- ctransformers
- transformers
- langchain
- langchain-community
- torch
- pypdf
- sentence-transformers
- faiss-cpu
- fastapi

---

## 📁 Project Structure

```
📦 RAG for Nha Trang University Admission Score Queries/
├── 📁 admission_score/
│   ├── admission_score2024.txt
│   └── admission_score2025.txt
│
├── 📁 vectorstores/
│   ├── index.faiss
│   ├── index.pkl
│   └── prepare_vector_db.py # File for converting personal data into a vector database
│
├── 📁 static/
│   └── style.css # CSS file
│
├── 📁 templates/
│   └── index.html # HTML file
│
├── qabot.py # Code file for chatbot Q&A
└── setup.txt


```
---
## 🚀 Installation & Quick Start
### 1. Clone the repository
```bash
git clone https://github.com/n-tan-adonis/RAG-for-Nha-Trang-University-Admission-Score-Queries
```
### 2. Install dependencies
```bash
pip install -r setup.txt
```
### 3. Convert personal data into a vector database
```bash
python vectorstores/prepare_vector_db.py
```
### 4. Chat with the Q&A bot
## 🔑 Set Up Gemini API Key  

Before running the chatbot, make sure to add your **Gemini API key** to the environment variables or directly in the code.  

### Option 1: Set environment variable (recommended)  
On macOS / Linux (Terminal):  
```bash
export GEMINI_API_KEY="your_actual_api_key"
```
On Windows (PowerShell):
```bash
setx GEMINI_API_KEY "your_actual_api_key"
```
### Option 2: Edit the code directly
Open the file **qabot.py**, and replace *"Your_API_Key"* with your actual Gemini API key in the following line:
```bash
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "Your_API_Key")
```
Example:
```bash
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBxxxxxx-your-real-key")
```
Once your API key is set, run the chatbot with:
```bash
python qabot.py
```
```bash
uvicorn qabot:app --reload
```

## 🎬 Demo

<p align="center">
  <img src="https://github.com/n-tan-adonis/RAG-for-Nha-Trang-University-Admission-Score-Queries/blob/main/result.png?raw=true" 
       alt="Result" width="80%" style="border-radius:10px;"/>
</p>

> 🖼️ *Illustration of admission score queries results in Nha Trang University.*


---
## 📬 Contact
- 📧 Email: nhattan13022003@gmail.com
- 🔗 LinkedIn: [Tan Truong](https://www.linkedin.com/in/truong-tan/)
