import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================
# 1. Cấu hình Gemini
# =====================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAnTDk0ompDK7douZVuUuGVwKk9hGmu5Ws")
genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-pro"]

def call_gemini(prompt: str, temperature: float = 0.1, max_output_tokens: int = 512) -> str:
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            )
            return getattr(resp, "text", str(resp))
        except Exception as e:
            print(f"Lỗi khi gọi Gemini với model={model_name}: {e}")
    return "Xin lỗi, tôi không thể trả lời câu hỏi này ngay lúc này."

# =====================================================
# 2. FAISS + Embeddings
# =====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}
)

try:
    vectorstore = FAISS.load_local(
        r"D:\ProjectPycharm\RAG\vectorstores",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ Vectorstore loaded successfully.")
except Exception as e:
    print(f"❌ Lỗi khi tải vectorstore: {e}")
    raise SystemExit(1)

# =====================================================
# 3. Prompt Template + Hàm truy vấn RAG
# =====================================================
prompt_template = """
Bạn là trợ lý AI chuyên nghiệp. Hãy trả lời câu hỏi dựa CHỦ YẾU trên nội dung sau đây.
Nếu thông tin không có trong nội dung này, hãy nói rõ bạn không biết.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}

Hãy trả lời một cách CHÍNH XÁC, RÕ RÀNG và chỉ dựa trên thông tin trên:
"""

def simple_rag_query(question: str):
    try:
        docs = retriever.invoke(question)
        if not docs:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.", []

        context = "\n".join([f"- {doc.page_content}" for doc in docs])
        full_prompt = prompt_template.format(context=context, question=question)
        answer = call_gemini(full_prompt)
        return answer, docs
    except Exception as e:
        print(f"Lỗi khi xử lý câu hỏi: {e}")
        return "Đã xảy ra lỗi khi xử lý câu hỏi.", []

# =====================================================
# 4. FastAPI App + Giao diện Web
# =====================================================
app = FastAPI(title="RAG Chatbot Interface")

# Khai báo thư mục chứa HTML và JS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QuestionRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
def query_rag(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    answer, docs = simple_rag_query(question)
    sources = [
        {"source": os.path.basename(doc.metadata.get("source", "Unknown")),
         "page": doc.metadata.get("page", "N/A")}
        for doc in docs
    ]
    return JSONResponse({"answer": answer, "sources": sources})
