
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

 
# Sidebar contents
with st.sidebar:
    st.title('PDF Chat App')
    st.markdown('''
    ## About
    A chat app that takes in PDF and answers user query based on the info in the PDF file uploded \n
    Made using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with by Kunal Bitey')
 
load_dotenv()
 
def main():
    st.header("PDF Q&A app")
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF here", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # Input user query
        query = st.text_input("Enter the question regarding your PDF :")
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=5)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            # with get_openai_callback() as cb:
            #     response = chain.run(input_documents=docs, question=query)
            #     print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()
