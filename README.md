# Langchain-Chatbot 🤖

This is a Streamlit-powered AI chatbot that utilizes the Llama 3.3 70B model via the Groq API to generate responses. The chatbot is built with LangChain, enabling seamless interaction with the LLM.

✨ Features

Llama 3.3 70B Model: Leverages the powerful Llama model for generating intelligent responses.

LangChain Integration: Uses LangChain’s prompt templates and chaining capabilities.

Streamlit UI: Provides a simple and interactive web-based interface.

.env Support: Securely loads API keys using python-dotenv.

🚀 How to Run

1. Clone this repository:

   git clone https://github.com/yourusername/llama-chatbot.git

   cd llama-chatbot

2. Install the required dependencies:

   pip install -r requirements.txt

3. Create a .env file and add your Groq API key:

   GROQ_API_KEY=your_groq_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key

4. Run the Streamlit app:
   
   streamlit run app.py

📌 Usage

Enter your question in the input field.

Click "Get Answer" to receive a response.

The chatbot will process the input using LangChain and return an AI-generated answer.

🛠️ Tech Stack

Python

Streamlit

LangChain

Groq API

Llama 3.3 70B Model

📜 License

This project is licensed under the MIT License.

Feel free to contribute and improve this chatbot! 🚀


