import streamlit as st
from dotenv import load_dotenv
import pickle
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# sidebar
with st.sidebar:
    st.title("🎉🎉 LLM Chat app")
    st.markdown('''
                ## About 
                This app app is an LLM Chatbot built using: 
                    - [Streamlit](https://streamlit.io)
                    - [LangChain](https://python.langchain.com/)
                    - [OpenAI](https://platform.openai.com/docs/models) LLM models
                    
                ''')
                
    add_vertical_space(5)
    st.write('made by Kunal Bitey')


def main():
    st.header("Chat with PDF 📁")
    
    load_dotenv()
    
    pdf = st.file_uploader("Upload your PDF here", type = 'pdf')
    
    if pdf:
        pdf_reader = PdfReader(pdf)
        
        text = ''
        
        for page in pdf_reader.pages:
            text = text + page.extract_text()
        
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text = text)
        
        
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                Vectorstore = pickle.load(f)
            
        else:
            # embeddings 
            embeddings = OpenAIEmbeddings()
            Vectorstore = FAISS.from_texts(chunks, embedding= embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(Vectorstore, f)
        
        # Accept query
        query = st.text_input("Enter the question regarding your PDF :")
        st.write(query)
        
        if query:
            docs = Vectorstore.similarity_search(query=query, k=3)
            
            llm = ChatOpenAI()
            chain = load_qa_chain(llm= llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question= query)
            st.write(response)
  
    
  
if __name__ == '__main__':
    main()
