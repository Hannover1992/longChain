import os
from global_var import chunk, overlap

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Split text into words
python_splitter = PythonCodeTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
embeddings = OpenAIEmbeddings()

db = Chroma(persist_directory='db', embedding_function=embeddings)

query = "Wie Schreibt man en Abstract?"
# docs = store.similarity_search(query)



# prompt = st.text_input('Niebo gwazdziste nademna, prawo moralne we mnie. A ty, czym jesteś?Jestem twoją wolnością. Jestem tym, co masz, czego się trzymasz, czym możesz zdecydować i dążyć do tego, co uważasz za słuszne. Jestem tym, co możesz zmienić i wpłynąć na życie innych. Jestem tym, co jest w twojej ręce.')
prompt = 'how ot wirte an abstract?'
search = db.similarity_search_with_score(prompt)
search.sort(key=lambda x: x[1], reverse=True)

for i in search:
    print("Content:" + i[0].page_content)
    print("Relevance:" + str(i[1]))
    print('------------------')


