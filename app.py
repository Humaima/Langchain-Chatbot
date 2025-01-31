import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableSequence


# Loading the API keys from the .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Initialize Llama 3.3 70B Model with Groq 
llama_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",  
    groq_api_key=GROQ_API_KEY
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["question"],  
    template="You are a helpful AI assistant. Answer this question: {question}"
)

# Creating a LangChain chain with the prompt and Llama model
llama_chain = RunnableSequence(prompt | llama_llm)

# Streamlit UI setup
st.set_page_config(page_title="Llama 3.3 70B AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Llama 3.3 70B AI Chatbot")
st.write("Ask me anything!")

# Input field for the user to ask a question
user_input = st.text_input("Your Question:", "")

# When the "Get Answer" button is pressed
if st.button("Get Answer"):
    if user_input:
        with st.spinner("Thinking..."):
            # Get the response from the Llama model using the chain
            response = llama_chain.invoke(user_input)
        st.success("Here's the answer:")
        st.write(response)
    else:
        st.warning("Please enter a question!")

# Footer section with some information about the app
st.markdown("---")
st.markdown("👨‍💻 Built with **LangChain, Llama 3.3 70B & Groq API**")
