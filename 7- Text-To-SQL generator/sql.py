import streamlit as st
import os
import sqlite3
import google.generativeai as genai
import pandas as pd

# Configure Gemini API
genai.configure(api_key="AIzaghsguhsdiufshfusdhfiugwiusghiufhsiqsY")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Define your prompt template
prompt = [
    """
You are a professional SQL assistant. Your job is to convert natural language questions into valid SQL queries.

The database name is STUDENT and the table structure is as follows:

Table: STUDENT
Columns:
- ID (INTEGER, PRIMARY KEY)
- NAME (TEXT)
- AGE (INTEGER)
- GENDER (TEXT: Male, Female, Other)
- CLASS (TEXT: Data Science, DevOps, Cybersecurity, AI, Web Development)
- SECTION (TEXT: A, B, C, D)
- GPA (REAL: Range 2.0 to 4.0)
- ENROLLMENT_DATE (DATE)
- EMAIL (TEXT)
- PHONE_NUMBER (TEXT)
- CITY (TEXT)

Here are some example questions and the SQL responses you should generate:

1. Question: How many students are in the database?
   SQL: SELECT COUNT(*) FROM STUDENT;

2. Question: Show me all students who enrolled after 2022.
   SQL: SELECT * FROM STUDENT WHERE ENROLLMENT_DATE > '2022-01-01';

3. Question: What is the average GPA of students in the Data Science class?
   SQL: SELECT AVG(GPA) FROM STUDENT WHERE CLASS = 'Data Science';

4. Question: List all students from section B who are older than 25.
   SQL: SELECT * FROM STUDENT WHERE SECTION = 'B' AND AGE > 25;

5. Question: Get the names and emails of female students in AI class with GPA above 3.5.
   SQL: SELECT NAME, EMAIL FROM STUDENT WHERE GENDER = 'Female' AND CLASS = 'AI' AND GPA > 3.5;

6. Question: What are the names of students living in Toronto?
   SQL: SELECT NAME FROM STUDENT WHERE CITY = 'Toronto';

Important Rules:
- Your output must be only the SQL query. Do not include any explanation or the words “SQL” or “Query”.
- Do NOT add triple backticks ``` or quotes around your output.
- Always match the table and column names exactly (case-sensitive).
- Always wrap string values in single quotes ('') when used in WHERE conditions.
"""
]

# Load Gemini model
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    response = model.generate_content([prompt[0], question])
    return response.text.strip()

# Run the SQL query
def read_sql_query(sql, db):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        conn.commit()
        conn.close()
        return rows, columns
    except Exception as e:
        return [(f"Error: {str(e)}",)], ["Error"]

# Streamlit UI
st.set_page_config(page_title="Natural Language to SQL Query App", layout="centered")
st.title("🎓 SQL Assistant with Gemini")
st.markdown("Type your question in natural language. This app will generate a SQL query, run it, and return the result.")

with st.form(key="query_form"):
    question = st.text_input("💬 Ask your question:")
    submit = st.form_submit_button("🔎 Get Result")

if submit and question:
    with st.spinner("Thinking... 🤖"):
        sql_query = get_gemini_response(question, prompt)
        rows, columns = read_sql_query(sql_query, "test_large.db")
        
        # Save history
        st.session_state.history.append({
            "question": question,
            "sql": sql_query,
            "result": rows,
            "columns": columns
        })

# Chat history display
if st.session_state.history:
    st.markdown("### 🕘 History")
    for entry in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(f"**You:** {entry['question']}")
        with st.chat_message("assistant"):
            st.markdown(f"**SQL:** `{entry['sql']}`")
            if "Error" in entry["columns"][0]:
                st.error(entry["result"][0][0])
            else:
                df = pd.DataFrame(entry["result"], columns=entry["columns"])
                st.dataframe(df, use_container_width=True)
