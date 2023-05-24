import os
from langchain.llms import OpenAI
import streamlit as st
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
# how to load text form .txt file
from langchain.document_loaders import PyPDFLoader



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.9)

prompt = st.text_input('Niebo gwazdziste nademna, prawo moralne we mnie. A ty, czym jesteś?Jestem twoją wolnością. Jestem tym, co masz, czego się trzymasz, czym możesz zdecydować i dążyć do tego, co uważasz za słuszne. Jestem tym, co możesz zmienić i wpłynąć na życie innych. Jestem tym, co jest w twojej ręce.')


def load_fiels():
    # data = st.file_uploader("Choose a file", type=['txt', 'pdf'])
    # loader = PyPDFLoader('test.pdf')
    # Open and read text from .txt file
    with open('all_txt.txt', 'r') as file:
        return file.read()


text_data = load_fiels()
store = Chroma.from_documents(text_data, collection_name='Wissenschaftliche Methoden')


if prompt:
    # response = llm(prompt)
    # st.write(response)

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt)
        st.write(search)