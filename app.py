from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import requests
import os
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

db = None
chain = None

"""#Use a Chain"""
custom_prompt_template = """Tôi muốn bạn đóng vai trò là Botchat PDF, được phát triển bởi nhóm sinh viên Khoa Công nghệ thông tin, Trường Đại học Mỏ - Địa chất.
Bạn sẽ đưa ra câu trả lời từ ngữ cảnh và câu hỏi dưới đây.
Câu trả lời của bạn phải đầy đủ, chính xác và mô tả chi tiết về nội dung câu hỏi của người dùng.
Giọng điệu của câu trả lời của bạn cần chuyên nghiệp.
Nếu bạn không biết câu trả lời, chỉ cần nói xin lỗi vì bạn không biết và đề cập đến việc yêu cầu người dùng mô tả chi tiết câu hỏi hơn, đừng cố bịa ra câu trả lời.
Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt

Context: {context}
Question: {question}

"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Khởi tạo FastAPI app
app = FastAPI()

# Định nghĩa request model
class Query(BaseModel):
    question: str

# Endpoint để xử lý PDF tải lên hoặc URL
@app.post("/upload_pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...), pdf_url: Optional[str] = Form(None)):
    if pdf_url:
        response = requests.get(pdf_url)
        filename = "downloaded_pdf.pdf"
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        filename = pdf_file.filename
        with open(filename, 'wb') as f:
            f.write(await pdf_file.read())

    # Xử lý PDF tại đây
    process_pdf(filename)  # Xử lý file PDF sau khi tải lên
    return {"filename": filename}

def process_pdf(filename):
    global db, chain
    pdf_loader = UnstructuredPDFLoader(filename)
    pdf_pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(pdf_pages)

    # Tạo Embeddings và Vectorstores
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = Chroma.from_documents(texts, hf_embeddings, persist_directory="db")
    
    # Cấu hình và sử dụng chain ở đây nếu cần
def initialize_chain():
    global chain
    if db is not None:
        chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={'prompt': prompt}
        )

#Model
"""Load Data & Model"""
os.environ["OPENAI_API_KEY"] = "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"
model = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

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
    with open('index.html', 'r', encoding='utf-8') as file:  # Thêm 'encoding='utf-8'
        html_content = file.read()
    return HTMLResponse(content=html_content)


# Hàm để chạy ứng dụng trên máy local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)