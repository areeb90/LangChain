from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import streamlit as st
import os
from dotenv import load_dotenv 

# ✅ Function to get response
def get_openai_response(prompt):

    # Load environment variables from .env file
    load_dotenv()

    # Ensure the OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0.7,
        openai_api_key= openai_api_key # Replace with your actual key
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ✅ Streamlit UI
st.set_page_config(page_title="Q&A Demo")
st.title("Q&A Chatbot")
st.write("Ask me anything about LangChain!")

user_question = st.text_input("Enter your question:")
if st.button("Submit"):
    if user_question:
        response = get_openai_response(user_question)
        st.write("Response:", response)
    else:
        st.warning("Please enter a question.")




# ===============================================================================================


#how to run this app
# Save this code in a file named app.py
# Open a terminal and navigate to the directory where app.py is located
# Run the following command:
# streamlit run app.py
# This will start a local server and open the app in your web browser
# You can then enter your question in the text input and click the "Submit" button to get a response from the OpenAI model.
# Make sure you have the required libraries installed:
# pip install streamlit langchain openai
# Also, ensure you have set your OpenAI API key in the environment variable OPENAI_API
# key or replace the line `os.environ["OPENAI_API_KEY"] = "your_openai_api_key"` with your actual API key.
# Note: The OpenAI API key should be kept secret and not shared publicly.