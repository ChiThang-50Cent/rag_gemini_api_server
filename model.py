import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

global qa_chain
qa_chain = None

API_KEY = os.getenv('API_KEY')

def init_vector_db(relative_path):
    loader = PyPDFLoader(relative_path)
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=900, chunk_overlap=30)
    # context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=API_KEY)
    vector_index = Chroma.from_documents(documents=texts, embedding=embeddings).as_retriever(search_kwargs={'k' : 20})

    return vector_index

def init_prompt_template():
    template_vi = """Bảng dưới là thông tin sản phẩm của một cửa hàng.
    Bao gồm Tên, Kho, Trạng thái, Giá, Danh mục.
    Sử dụng các thông tin đó để trả lời câu hỏi của người dùng.
    Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết. Không đưa thông tin giả.
    Tất cả câu trả lời đều phải dùng tiếng Việt.
    {context}
    Câu hỏi: {question}
    Câu trả lời:"""

    # template_en = """The information below is about the details of a store's products.
    # Use that information to answer user questions.
    # If you don't know the answer, just say you don't know. Do not give false information.
    # All answers must be in Vietnamese.
    # {context}
    # Question: {question}
    # Answer:"""

    qa_prompt = PromptTemplate.from_template(template_vi)

    return qa_prompt

def init_model():
    global qa_chain

    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=API_KEY,
                             temperature=1,convert_system_message_to_human=True)

    vector_index = init_vector_db("./data/products_20240309185935.pdf")
    qa_prompt = init_prompt_template()

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        chain_type_kwargs={"prompt": qa_prompt}
    )

def Q_A(question):
    global qa_chain

    result = qa_chain.invoke(question)

    return result["result"]