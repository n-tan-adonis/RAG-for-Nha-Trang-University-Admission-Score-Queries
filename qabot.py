import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# 1) Cấu hình Gemini
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "Your_API_Key")
genai.configure(api_key=GEMINI_API_KEY)

# Bạn có thể đổi thứ tự ưu tiên model ở đây
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-pro",
]

def call_gemini(prompt: str, temperature: float = 0.1, max_output_tokens: int = 512) -> str:
    """
    Gọi Gemini: thử lần lượt các model trong GEMINI_MODELS.
    Nếu model đầu 404 (không tồn tại/không hỗ trợ), tự động thử model tiếp theo.
    """
    last_err = None
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
            # SDK mới trả về .text nếu là trả lời văn bản
            return getattr(resp, "text", str(resp))
        except Exception as e:
            last_err = e
            print(f"Lỗi khi gọi Gemini với model={model_name}: {e}")
            continue
    return "Xin lỗi, tôi không thể trả lời câu hỏi này ngay lúc này."

# =========================
# 2) Chuẩn bị FAISS + Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}  # đổi "cuda" nếu có GPU
)

try:
    vectorstore = FAISS.load_local(
        r"D:\ProjectPycharm\RAG\vectorstores",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Đã tải vectorstore thành công")
except Exception as e:
    print(f"Lỗi khi tải vectorstore: {e}")
    raise SystemExit(1)

# =========================
# 3) Prompt template
# =========================
prompt_template = """
Bạn là trợ lý AI chuyên nghiệp. Hãy trả lời câu hỏi dựa CHỦ YẾU trên nội dung sau đây.
Nếu thông tin không có trong nội dung này, hãy nói rõ bạn không biết.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}

Hãy trả lời một cách CHÍNH XÁC, RÕ RÀNG và chỉ dựa trên thông tin trên:
"""

# =========================
# 4) Hàm RAG đơn giản
# =========================
def simple_rag_query(question: str):
    try:
        docs = retriever.invoke(question)
        if not docs:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.", []

        # Ghép context từ các chunk
        context = "\n".join([f"- {doc.page_content}" for doc in docs])
        full_prompt = prompt_template.format(context=context, question=question)

        answer = call_gemini(full_prompt)
        return answer, docs
    except Exception as e:
        print(f"Lỗi khi truy xuất tài liệu: {e}")
        return "Có lỗi xảy ra khi xử lý câu hỏi.", []

# =========================
# 5) Chạy thử
# =========================
if __name__ == "__main__":
    question = "Điểm chuẩn ngành Công Nghệ Thông Tin 2025"
    print(f"Đang xử lý câu hỏi: {question}")
    answer, source_docs = simple_rag_query(question)

    print("\n" + "=" * 50)
    print("CÂU TRẢ LỜI:")
    print("=" * 50)
    print(answer)

    if source_docs:
        print("\n" + "=" * 50)
        print("TÀI LIỆU THAM KHẢO:")
        print("=" * 50)
        for i, doc in enumerate(source_docs, start=1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            print(f"{i}. {os.path.basename(source)} - Trang {page}")
            # In TOÀN BỘ chunk, bỏ dấu "..." rút gọn
            print("   Nội dung:")
            print(doc.page_content)  # <- in full
            print()
