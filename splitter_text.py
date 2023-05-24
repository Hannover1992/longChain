from langchain.text_splitter import PythonCodeTextSplitter


def load_data(filename, chunk_size=5000):
    with open(filename, 'r') as file:
        text = file.read()
    return text
    # # Create a list of Document objects, each containing a chunk of text
    # return [Document(text[i:i + chunk_size]) for i in range(0, len(text), chunk_size)]


text_data_chunks = load_data('all_txt.txt')
print(text_data_chunks)
python_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = python_splitter.create_documents([text_data_chunks])
print(python_splitter)