import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split
from src.helper import download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing in .env")

print("📄 Loading PDF files from Data/ ...")
extracted_data = load_pdf_file(data_path="Data/")
print(f"✅ Loaded {len(extracted_data)} documents.")

print("✂️ Splitting text into chunks ...")
text_chunks = text_split(extracted_data)
print(f"✅ Created {len(text_chunks)} text chunks.")

print("🧠 Loading Hugging Face embeddings ...")
embeddings = download_hugging_face_embeddings()

print("🔗 Connecting to Pinecone ...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# --- Commented out index creation/upload because already done ---
# if index_name not in [idx.name for idx in pc.list_indexes()]:
#     print(f"📦 Creating Pinecone index '{index_name}' ...")
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
#     print("✅ Index created successfully!")
# else:
#     print(f"ℹ️ Index '{index_name}' already exists.")

# print("⬆️ Uploading documents to Pinecone ...")
# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     index_name=index_name,
#     embedding=embeddings
# )
# print("🎯 Pinecone index setup complete and data uploaded.")

print("✅ Pinecone index setup skipped, assuming index and data exist.")
