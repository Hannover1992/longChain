from langchain.text_splitter import PythonCodeTextSplitter
with open('all_txt.txt', 'r') as file:
    text = file.read()

# Split text into words
python_splitter = PythonCodeTextSplitter(chunk_size=30, chunk_overlap=0)
docs = python_splitter.create_documents([text])
print(docs)

python_splitter.split_text(text)

