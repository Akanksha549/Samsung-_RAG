import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Samsung RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main{
    padding-top:1rem;
}
.stChatMessage{
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🤖 Samsung Washing Machine RAG Chatbot")
st.write("Ask anything from the Samsung Washing Machine Manual.")

# -----------------------------
# API KEY
# -----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# Cache Vector Database
# -----------------------------
@st.cache_resource
def load_rag():

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    loader = UnstructuredHTMLLoader("1.html")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":3}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
You are a Samsung Washing Machine AI Assistant.

Answer ONLY using the supplied manual.

If the answer is unavailable, reply:

"I don't know based on the provided manual."

Context:
{context}

Question:
{question}

Answer:
""")

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain


rag = load_rag()

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------
question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append(
        {"role":"user","content":question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching manual..."):
        answer = rag.invoke(question).content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )
