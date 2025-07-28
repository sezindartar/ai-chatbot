from pdf_utils import get_pdf_text
from embedding_utils import get_vectorstore, get_text_chunks
from qa_chain import get_conversational_chain
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings




load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


if not openai_api_key:
    st.error("Please set OPENAI_API_KEY environment variable.")
    st.stop()



st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")
st.title("PDF Chatbot")
st.write("Please upload your PDF files.")



if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



with st.sidebar:
    st.header("📁 Upload PDFs")
    pdf_docs = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)



if pdf_docs:
    raw_text = get_pdf_text(pdf_docs)
    st.success("✅ Text extracted from PDFs.")
    st.write(raw_text[:1000])  # Preview


    text_chunks = get_text_chunks(raw_text)
    st.success("✅ Text split into chunks.")

 
    vectorstore = get_vectorstore(text_chunks)
    st.success("✅ Vector store created.")

    st.subheader("❓ Ask a question")
    user_question = st.text_input("Enter your question")

    if user_question:
        chain = get_conversational_chain(vectorstore=vectorstore)
        result = chain({"query": user_question})
        answer = result.get("result")

        st.subheader("📢 Answer:")
        st.write(answer)

        with st.expander("📚 Source Documents"):
            for doc in result["source_documents"]:
                st.markdown(doc.page_content)


        st.session_state.chat_history.append((user_question, answer))


    if st.session_state.chat_history:
        st.subheader("💬 Chat History:")
        for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}:** {question}")
            st.markdown(f"**A{i}:** {answer}")
