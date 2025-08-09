import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from typing import Optional, List

import google.generativeai as genai  # Gemini SDK
from pinecone import Pinecone  # Latest Pinecone SDK
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms.base import LLM

# 1. Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
INDEX_NAME = "medical-chatbot"

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Set PINECONE_API_KEY and GOOGLE_API_KEY in .env")

# 2. Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# 3. Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
pinecone_index = pc.Index(INDEX_NAME)

# 4. Setup embeddings and vector store
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity",
                                   search_kwargs={"k": 3})

# 5. Define system prompt
system_prompt = (
    "You are a helpful medical assistant for question-answering tasks. "
    "Always start your answer with phrases like 'From the information I have,"
    "or 'According to the data I’ve seen,' followed by the concise answer. "
    "Only use the retrieved context to answer. "
    "If you don't know, say 'I don’t have information about it.' "
    "Do not say 'as per given' or 'Based on the provided text'. "
    "Be clear and concise.\n\n{context}"
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# 6. GeminiLLM Class
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"
    temperature: float = 0.4
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt_text: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            prompt_text,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
        )
        return response.text


llm = GeminiLLM()

# 7. Build Retrieval Chain
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 8. Flask App
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def get_response():
    user_input = request.form.get("msg")
    if not user_input:
        return "Please enter a question."

    print(f"User: {user_input}")
    response = rag_chain.invoke({"input": user_input})

    # Force "I don’t have information about it." if no answer
    answer = response.get("answer") or response.get("output_text") or ""
    if not answer.strip() or "does not contain information" in answer.lower():
        answer = "I don’t have information about it."

    print(f"Bot: {answer}")
    return answer


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
