import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Khai báo đường dẫn
txt_data_path = r"D:\ProjectPycharm\RAG\admission_score"   # Thư mục chứa .txt
vector_db_path = r"D:\ProjectPycharm\RAG\vectorstores"

def create_db_from_txt():
    all_docs = []

    # Quét tất cả file .txt trong thư mục
    for file in os.listdir(txt_data_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(txt_data_path, file), encoding="utf-8")
            all_docs.extend(loader.load())

    # Chia nhỏ văn bản: mỗi chunk chứa 1 ngành hoặc 1 block thông tin
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Mỗi chunk khoảng 400 ký tự, đủ chứa 1 ngành
        chunk_overlap=20,     # Giữ 20 ký tự chồng lặp tránh mất ngữ cảnh
        separators=["\n\n", "\n", " ", ""]  # Tách ưu tiên theo xuống dòng
    )
    chunks = text_splitter.split_documents(all_docs)

    # Sử dụng mô hình embedding BAAI/bge-small-en-v1.5
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}  # Đổi thành "cuda" nếu có GPU
    )

    # Tạo vector database
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    print(f"Tạo xong Vector DB với {len(chunks)} chunks")
    return db

# Gọi hàm
create_db_from_txt()
