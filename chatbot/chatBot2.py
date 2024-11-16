import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = "sk-proj-"

st.session_state.messages = []
st.session_state.chat_history = []


def get_pdf_text(pdf_doc):
    loader = PyPDFLoader(pdf_doc.name)
    docs = loader.load()
    return docs


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = splitter.split_documents(text)
    return chunks


def get_vector_store(chunks):
    #embeddings = chroma.from_documents(documents=chunks, embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return embeddings

def generate_rag_chain(embeddings, prompt):
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=1000,
        model_name="gpt-4o-mini"
    )

    #def generate_rag_chain(embeddings):
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retriever = embeddings.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation generate a search query to look up in order to get information")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )
    return retrieval_chain

def generate_response(embeddings):
    prompt_to_user = "How may I help you?"

    if "messages" not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', 'content': prompt_to_user}]
        st.session_state.chat_history = [AIMessage(content=prompt_to_user)]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #def generate_response(embeddings):
    if prompt1 := st.chat_input("Ask something..."):
        st.chat_message("user").markdown(prompt1)
        st.session_state.messages.append({'role': 'user', 'content': prompt1})
        st.session_state.chat_history.append(HumanMessage(content=prompt1))

        chain = generate_rag_chain(embeddings, prompt1)

        response = process_chat(chain, prompt1)

        #st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.messages.append({'role': 'assistant', 'content': response})

        with st.chat_message("assistant"):
            st.markdown(response)
        #st.session_state.messages.append({'role': 'assistant', 'content': response})


def process_chat(chain, user_question):
    response = chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_question
    })

    print("process_chat response = ", response)
    return response["answer"]


def main():
    st.set_page_config(page_title="LLM App", page_icon="&")
    st.title("LLM App")

    pdf = st.file_uploader(label="Upload your file")
    if pdf is not None:
        with open(pdf.name, mode='wb') as w:
            w.write(pdf.getvalue())

        #if pdf is not None:
        docs = get_pdf_text(pdf)
        chunks = get_text_chunks(docs)
        embeddings = get_vector_store(chunks)
        generate_response(embeddings)


if __name__ == '__main__':
    main()
