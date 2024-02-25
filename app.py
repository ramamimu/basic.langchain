import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import openai, huggingface
from langchain.vectorstores import faiss
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = openai.OpenAIEmbeddings()
    embeddings = huggingface.HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_texts(
        texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_cain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_cain


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Simple Chatbot with Langchain", page_icon=":books:")
    st.header("Chat with multiple PDF inputs")
    st.text_input("Ask a question about your document")

    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            st.spinner("Processing")

            # get the pdf text
            raw_text = get_pdf_text(pdf_docs)
            st.write(raw_text)

            # get the text chunk
            text_chunks = get_text_chunks(raw_text)
            st.write(text_chunks)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # conversation chain
            # st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
