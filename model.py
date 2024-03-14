import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

global qa_chain
qa_chain = None

def init_vector_db(relative_path):
    loader = PyPDFLoader(relative_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    # context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key='AIzaSyAbFiGnRBz_CIZhWbWLZ8gqeXgQvMDtb7o')
    vector_index = Chroma.from_documents(documents=texts, embedding=embeddings).as_retriever()

    return vector_index

def init_prompt_template():
    template = """Sử dụng các thông tin kèm theo để trả lời câu hỏi của người dùng.
    Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết.
    Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
    {context}
    Câu hỏi: {question}
    Câu trả lời:"""

    qa_prompt = PromptTemplate.from_template(template)

    return qa_prompt

def init_model():
    global qa_chain

    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key='AIzaSyAbFiGnRBz_CIZhWbWLZ8gqeXgQvMDtb7o',
                             temperature=0.5,convert_system_message_to_human=True)

    vector_index = init_vector_db("./data/products_20240309185935.pdf")
    qa_prompt = init_prompt_template()

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

def Q_A(question):
    global qa_chain

    result = qa_chain.invoke(question)

    return result["result"]