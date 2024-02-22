import os
from dotenv import load_dotenv
import google.generativeai as gen_ai
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

DB_PATH='db/'

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")




custom_prompt_template = """Use the following pieces of information to answer the user's question.
dont add the bullet marks,numbering and other signs in the answer. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:

"""

# setting prompt
def set_prompt():
     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
     return prompt
 
# creating qa 
def qa_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})
    
    db = FAISS.load_local(DB_PATH, embeddings)
    
    prompt=set_prompt()
    qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                               retriever=db.as_retriever(search_kwargs={'k': 4}),
                               return_source_documents=False, 
                               chain_type_kwargs={"prompt":prompt})
    return qa

# chain=qa_model()
# prompt="what is the legal procedure to get building permit"
# ans=chain.invoke({'query':prompt})
# print(ans)


# stream_lit front-end
st.title("welcome")
with st.sidebar:
    st.title('🤖💬 KMBR law assistant')
  
    st.success('Proceed to entering your prompt message!', icon='👉')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("ask your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        chain=qa_model()
        res=chain.invoke({'query':prompt})
        answer=res["result"]
        
        message_placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})



