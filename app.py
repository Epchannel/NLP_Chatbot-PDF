from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pyngrok import ngrok
import nest_asyncio
import uvicorn
import os
import textwrap
import chromadb
import langchain
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, YoutubeLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from pdf2image import convert_from_path

"""Load Data & Model"""
os.environ["OPENAI_API_KEY"] = "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"
model = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
images = convert_from_path("Quy dinh 768.pdf", dpi=88)
len(images)
images[0]

"""Load pdf"""
pdf_loader = UnstructuredPDFLoader("Quy dinh 768.pdf")
pdf_pages = pdf_loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(pdf_pages)
len(texts)

"""Create Embeddings & Vectorstores"""
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma.from_documents(texts, hf_embeddings, persist_directory="db")




"""#Use a Chain"""
custom_prompt_template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng.
Bạn là Chatbot để hỗ trợ giải đáp thắc mắc cho sinh viên HUMG.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt
Context: {context}
Question: {question}
"""
#Framework Langchain
from langchain import PromptTemplate
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

prompt = set_custom_prompt()
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={'prompt': prompt}
)

from fastapi.responses import HTMLResponse
from pyngrok import ngrok
from fastapi import FastAPI, Form, HTTPException

#Set auth token ngrok
ngrok.set_auth_token("2RSYEiqVjwKFD4C46bBHnlt0fRA_5yaKGD8nqzx2TCiapqnM6")

# Khởi tạo FastAPI app
app = FastAPI()

# Định nghĩa request model
class Query(BaseModel):
    question: str

# API endpoint
@app.post("/ask")
async def ask_question(query: Query):
    try:
        response = chain.run(query.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open('chatbot_form.html', 'r', encoding='utf-8') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


# Hàm để chạy ứng dụng
def run_app():
    ngrok_tunnel = ngrok.connect(8000)
    print('URL công khai:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

#@title # ▶️ 4. Chạy ứng dụng
# Gọi hàm run_app()
run_app()