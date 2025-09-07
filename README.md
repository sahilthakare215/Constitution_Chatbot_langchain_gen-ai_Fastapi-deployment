NOTE:  the link is slow because its a free version of render so it will take 1-2 minutes to load but later your experience with the UI will be smooth.i could have used vercel or netlify but both show so much of errors while working with fastapi so i chose render and also streamlit deployment is present in another repo.

https://constitutio-whgc0z.streamlit.app/


# Constitution Chatbot API

## Description
The Constitution Chatbot API is a FastAPI application that allows users to query the Indian Constitution using a Retrieval-Augmented Generation (RAG) approach. It leverages Google Generative AI for natural language processing and provides a user-friendly interface for accessing constitutional information.

## Features
- Query the Indian Constitution
- Retrieve answers with supporting articles
- Health check endpoint
- Multi-turn conversation support

## Installation

### Prerequisites
- Python 3.11 or higher
- Pip

### Clone the Repository
```bash
git clone <repository-url>
cd constitution-chatbot
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file in the root directory and add the following variables:
```
GOOGLE_API_KEY=your_google_api_key_here
PDF_PATH=documents/20240716890312078.pdf
GEMINI_MODEL=gemini-1.5-flash-latest
EMBEDDING_MODEL=models/embedding-001
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SEMANTIC_SIMILARITY_THRESHOLD=0.75
HOST=0.0.0.0
PORT=8000
DEBUG=False
CORS_ORIGINS=*
```

## Running the Application
To start the application, run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### 1. Root Endpoint
- **GET** `/`
  - Serves the frontend interface.

### 2. API Information
- **GET** `/api`
  - Returns API information.

### 3. Health Check
- **GET** `/health`
  - Returns the health status of the service.

### 4. Ask a Question
- **POST** `/ask`
  - Request body:
    ```json
    {
      "question": "What is the Constitution of India?",
      "conversation_id": "12345"
    }
    ```
  - Returns the answer, source, supporting articles, and semantic similarity.

### 5. Multi-turn Conversation
- **POST** `/conversation`
  - Request body:
    ```json
    {
      "messages": [{"content": "What is the Constitution?"}],
      "conversation_id": "12345"
    }
    ```
  - Returns the response for the conversation.

### 6. Statistics
- **GET** `/stats`
  - Returns service statistics and fallback logs.

## License
This project is licensed under the MIT License.
