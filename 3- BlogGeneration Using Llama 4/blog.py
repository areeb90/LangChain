#  I used a Python langchain app to create blogs. The app allows me to input a topic, specify the number of words, and choose the target audience. With just a click, the app generates a blog on the given topic in the specified number of words for the selected audience. It's a great tool for researchers, data scientists, and common people who want to quickly generate quality content.



import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.schema import HumanMessage


def getLLamaresponse(input_text, no_words, blog_style):
    prompt = PromptTemplate(
        input_variables=["input_text", "no_words", "blog_style"],
        template="Write a blog on {input_text} in {no_words} words for {blog_style}."
    )

        # Initialize ChatOpenAI model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0.7,
        openai_api_key="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"
    )
    
    # Simulating a response from Llama 4
    response = llm([
        HumanMessage(content=prompt.format(
            input_text=input_text,
            no_words=no_words,
            blog_style=blog_style
        ))
    ])
    return response.content

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))

