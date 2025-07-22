import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

# Import necessary modules for document loading and vector store
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check for API key
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Add it to your .env file.")

# Initialize ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Set up memory for conversation history
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Set up LangChain ConversationChain with memory
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# Prompt Template (to keep assistant behavior consistent)
prompt_template = ChatPromptTemplate.from_template(
    "You are a helpful assistant specialized in LangChain and LLMs.\n\n{question}"
)

# Streamlit UI
st.set_page_config(page_title="LangChain Chatbot", layout="centered")
st.title("🧠 LangChain Assistant")
st.markdown("Ask me anything about **LangChain, LLMs, or OpenAI tools**!")

# Chat input
user_question = st.chat_input("Type your question here...")

# Maintain full chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


st.sidebar.header("📄 Document Q&A")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

use_pdf_qa = False
retrieval_chain = None

if uploaded_file:
    with st.spinner("Processing document..."):
        # Save uploaded PDF temporarily
        temp_path = os.path.join("temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load and split the PDF
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(pages)

        # Create vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = FAISS.from_documents(texts, embeddings)

        # Create a retrieval chain
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        use_pdf_qa = True
        st.sidebar.success("PDF processed! Now ask questions in chat.")



if user_question:
    try:
        if use_pdf_qa and retrieval_chain:
            result = retrieval_chain.invoke({"query": user_question})
            bot_reply = result["result"]
        else:
            prompt = prompt_template.format_messages(question=user_question)
            response = llm.invoke(prompt)
            bot_reply = response.content
    except Exception as e:
        bot_reply = f"❌ Error: {str(e)}"

    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("bot", bot_reply))


# Display chat
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


# =============================================================== 

# from langchain.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage
# import streamlit as st
# import os
# from dotenv import load_dotenv 

# # ✅ Function to get response
# def get_openai_response(prompt):

#     # Load environment variables from .env file
#     load_dotenv()

#     # Ensure the OpenAI API key is set
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    

#     llm = ChatOpenAI(
#         model="gpt-3.5-turbo-0125",
#         temperature=0.7,
#         openai_api_key= openai_api_key # Replace with your actual key
#     )
    
#     response = llm.invoke([HumanMessage(content=prompt)])
#     return response.content

# # ✅ Streamlit UI
# st.set_page_config(page_title="Q&A Demo")
# st.title("Q&A Chatbot")
# st.write("Ask me anything about LangChain!")

# user_question = st.text_input("Enter your question:")
# if st.button("Submit"):
#     if user_question:
#         response = get_openai_response(user_question)
#         st.write("Response:", response)
#     else:
#         st.warning("Please enter a question.")




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