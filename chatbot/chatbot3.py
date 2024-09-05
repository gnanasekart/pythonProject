# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.chat_models import ChatOpenAI
#
# OPENAI_API_KEY = "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"
#
# # Initialize Streamlit session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# # Upload PDF files
# st.header("My First Chatbot")
#
# with st.sidebar:
#     st.title("Your Documents")
#     file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")
#
# # Extract the text
# if file is not None:
#     pdf_reader = PdfReader(file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#
#     # Break it into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators="\n",
#         chunk_size=1000,
#         chunk_overlap=150,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#
#     # Generating embeddings
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#
#     # Creating vector store - FAISS
#     vector_store = FAISS.from_texts(chunks, embeddings)
#
#     # Get user question
#     user_question = st.text_input("Type your question here")
#
#     if user_question:
#         # Store the user question in the chat history
#         st.session_state.chat_history.append({"role": "user", "content": user_question})
#
#         # Do similarity search
#         match = vector_store.similarity_search(user_question)
#
#         # Define LLM
#         llm = ChatOpenAI(
#             openai_api_key=OPENAI_API_KEY,
#             temperature=0,
#             max_tokens=1000,
#             model_name="gpt-4o-mini"
#         )
#
#         # RAG process: Use the chat history and the retrieved documents
#         context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in st.session_state.chat_history])
#         documents_context = "\n".join([doc.page_content for doc in match])
#
#         full_context = context + "\n\nRetrieved Document:\n" + documents_context
#
#         # Chain of event -> take the context, get relevant answers, pass it to the LLM, generate the output
#         chain = load_qa_chain(llm, chain_type="stuff")
#         response = chain.run(input_documents=match, question=full_context)
#
#         # Store the response in chat history
#         st.session_state.chat_history.append({"role": "assistant", "content": response})
#
#         # Display the response
#         st.write(response)


import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "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"

# Initialize Streamlit session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF files
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Display chat history
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**{chat['role'].capitalize()}:** {chat['content']}")

    # Get user question
    user_question = st.text_input("Type your question here")

    if user_question:
        # Store the user question in the chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Do similarity search
        match = vector_store.similarity_search(user_question)

        # Define LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-4o-mini"
        )

        # RAG process: Use the chat history and the retrieved documents
        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in st.session_state.chat_history])
        documents_context = "\n".join([doc.page_content for doc in match])

        full_context = context + "\n\nRetrieved Document:\n" + documents_context

        # Chain of event -> take the context, get relevant answers, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=full_context)

        # Store the response in chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display the response
        st.write(response)

