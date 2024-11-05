# Building Intelligent Chatbots

This repository demonstrates how to build chatbots with different levels of memory capabilities using LangChain, Pinecone, and Streamlit. The project shows the evolution from a stateless chatbot to one with permanent memory storage.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://python.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Latest-yellow.svg)](https://www.pinecone.io/)

## 🌟 Features

- Three levels of chatbot implementation:
  - Level 1: Stateless bot (No Memory) 🐠
  - Level 2: Temporary memory using session state 🐹
  - Level 3: Permanent memory using Pinecone vector storage 🐘
- Clean, modular code structure
- Built with modern AI tools and frameworks
- Complete with error handling and best practices

## 🛠️ Prerequisites

Before you begin, ensure you have:
- Python 3.9 or higher
- API keys for:
  - Pinecone
  - OpenAI (for embeddings)
  - Groq (for LLM)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/spandan114/building-intelligent-chatbots.git
cd building-intelligent-chatbots
```

### 2. Set Up Virtual Environment

On Windows:
```bash
# Create virtual environment
python -m venv chatbotenv

# Activate virtual environment
chatbotenv\Scripts\activate
```

On macOS/Linux:
```bash
# Create virtual environment
python -m venv chatbotenv

# Activate virtual environment
source chatbotenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Set Up Pinecone

1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io/)
2. Create a new index with:
   - Name: chat-memory
   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: cosine

### 6. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### 7. Deactivate Virtual Environment When Done

```bash
deactivate
```

## 📁 Project Structure

```
building-intelligent-chatbots/
├── bot_with_pinecone_memory.py  # Main application with Pinecone 
├── bot_without_memory.py         # Basic bot implementation
├── bot_with_temporary_memory.py # Bot with session state memory
├── requirements.txt           # Project dependencies
├── .env                      # Environment variables (create this)
└── README.md                 # This file
```

## 🔧 Available Implementations

1. **Basic Bot** (`bot_without_memory.py`):
   ```bash
   streamlit run bot_without_memory.py
   ```

2. **Temporary Memory Bot** (`bot_with_temporary_memory.py`):
   ```bash
   streamlit run bot_with_temporary_memory.py
   ```

3. **Permanent Memory Bot** (`bot_with_pinecone_memory.py`):
   ```bash
   streamlit run bot_with_pinecone_memory.py
   ```

## 📝 Requirements

```
langchain
langchain-groq
langchain-pinecone
langchain-huggingface
langchain-postgres
langchain-community
langchain-core
streamlit
python-dotenv
pinecone-client
psycopg2-binary
sentence-transformers
openai
tiktoken
```

## 🐛 Troubleshooting

1. **ModuleNotFoundError**:
   - Ensure your virtual environment is activated
   - Verify all dependencies are installed: `pip install -r requirements.txt`

2. **API Key Errors**:
   - Check if your `.env` file exists and contains valid API keys
   - Ensure the API keys have proper permissions

3. **Pinecone Errors**:
   - Verify your index is created with correct specifications
   - Check if your API key has the necessary permissions


## 👏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the amazing framework
- [Pinecone](https://www.pinecone.io/) for vector storage capabilities
- [Streamlit](https://streamlit.io/) for the web interface

## ✨ Star the Repository

If you found this project helpful, please consider giving it a star! It helps others discover the project and motivates us to create more content.

## 📧 Contact

For questions and feedback, please open an issue in the GitHub repository.

Happy coding! 🚀