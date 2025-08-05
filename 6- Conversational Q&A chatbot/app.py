## Conversational Q&A Chatbot
import streamlit as st

from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain.chat_models import ChatOpenAI

## Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")


chat=ChatOpenAI(openai_api_key= "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",
                model_name="gpt-3.5-turbo",)


if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[
        SystemMessage(content="Yor are a comedian AI assitant")
    ]

def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    response=chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=response.content))
    return response.content


input=st.text_input("Input: ",key="input")
response=get_chatmodel_response(input)

submit=st.button("Ask the question")




## If ask button is clicked

if submit:
    st.subheader("The Response is")
    st.write(response)