🎙️ Customer Support Voice Agent

Welcome to my AI Customer Support Voice Agent! 🚀
This project transforms documentation into an intelligent, voice-enabled AI assistant using Retrieval-Augmented Generation (RAG).
It crawls documentation websites, builds a semantic knowledge base, retrieves relevant information using vector search, generates accurate responses using Groq LLM, and converts answers into speech.
__________________________________________________________________________________________________________________________________________________________________
📌 Features

Documentation crawling using Firecrawl
Semantic search using Qdrant vector database
Local embedding generation with FastEmbed
Intelligent response generation using Groq LLM
Voice-enabled answers using gTTS
Clean and interactive Streamlit interface
Multiple language support for TTS
Adjustable speech speed option
Fully OpenAI-free architecture
__________________________________________________________________________________________________________________________________________________________________
🛠️ Technologies Used

Python – Core programming language
Streamlit – Interactive web interface
Groq – Large Language Model inference
Qdrant – Vector database
FastEmbed – Local embedding generation
Firecrawl – Documentation crawling
gTTS – Text-to-speech conversion
python-dotenv – Environment variable management
asyncio – Asynchronous processing
__________________________________________________________________________________________________________________________________________________________________
📂 Project Structure
customer-support-voice-agent/
│-- app.py
│-- requirements.txt
│-- README.md
│-- .env
🏗️ Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/yourusername/customer-support-voice-agent.git
cd customer-support-voice-agent
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Configure Environment Variables

Create a .env file in the root directory:

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
GROQ_API_KEY=your_groq_api_key
4️⃣ Run the Application
streamlit run app.py

Open in your browser:

http://localhost:8501
__________________________________________________________________________________________________________________________________________________________________
🚀 How It Works

Enter documentation URL in sidebar.
System crawls and processes documentation.
Content is converted into vector embeddings.
Embeddings are stored in Qdrant.
User asks a question.
Relevant documentation is retrieved using semantic search.
Groq LLM generates context-aware response.
gTTS converts response into speech.
User receives both text and audio output.
__________________________________________________________________________________________________________________________________________________________________
🎯 Use Cases

API Documentation Assistant
SaaS Customer Support Bot
Developer Helpdesk Assistant
Internal Knowledge Base Search
AI Voice Helpdesk System
__________________________________________________________________________________________________________________________________________________________________
🚀 Future Improvements

Add document chunking for better accuracy
Display source citations
Implement streaming responses
Add authentication system
Deploy to cloud platforms
__________________________________________________________________________________________________________________________________________________________________
🎯 Contributions

Feel free to fork this repository and submit pull requests.
If you find any issues, report them in the Issues section.
