import azure.functions as func
import logging
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from openai import OpenAI
from docx import Document as DocxDocument
import os
import json
from azure.storage.blob import BlobServiceClient

class KnowledgeBase:
    def __init__(self, text_data):
        texts=self.text_split(text_data)
        self.documents = self.convert_texts_to_documents(texts)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key= os.environ["OPENAI_API_KEY"])
        self.retriever = None

    @staticmethod
    def text_split(extracted_data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size =200, chunk_overlap =50,separators=["\n"])
        text_chunks = text_splitter.split_text(extracted_data)
        return text_chunks

    def build_retriever(self,save=True):
        store = FAISS.from_documents(self.documents, self.embeddings)
        if save:
            store.save_local("faiss_index")
        self.retriever =store.as_retriever()

    @staticmethod
    def convert_texts_to_documents(texts):
        documents = [Document(page_content=text) for text in texts]
        return documents

class VirtualAssistant:
    def __init__(self, retriever, general=True, llm_model="gpt-3.5-turbo"):
        self.retriever = retriever
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.general = general

    def answer_query(self, query, context):
        custom_prompt_template = (
            f"Vous êtes une assistante virtuelle spécialisée pour un cabinet de radiologie médicale.\n"
            f"Voici des informations pertinentes extraites de notre base de connaissances :\n{context}\n\n"
            f"Question du patient : {query}\n\n"
            f"Répondez uniquement à la question posée, de manière claire et concise."
        )

        completion = self.client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": custom_prompt_template},
                {"role": "user", "content": query}
            ]
        )
        return completion.choices[0].message.content

class RAG_Azure:
    def __init__(self , llm_model="gpt-3.5-turbo"):
        if os.path.exists("faiss_index"):
            self.knowledge_base = KnowledgeBase('')
            self.knowledge_base.retriever = FAISS.load_local("faiss_index", self.knowledge_base.embeddings).as_retriever()
        else:
            documents = self.load_files_contents('data')
            self.knowledge_base = KnowledgeBase(documents)
            self.knowledge_base.build_retriever()

        self.assistant = VirtualAssistant(retriever=self.knowledge_base.retriever,general=True,llm_model=llm_model)

    @staticmethod
    def load_files_contents(data_path):
        concatenated_content = ""
        for filename in os.listdir(data_path):
            filepath = os.path.join(data_path, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                concatenated_content += "\n".join([doc.page_content for doc in documents]) + "\n"
            elif filename.endswith(".txt"):
                with open(filepath, 'r', encoding='utf-8') as file:
                    concatenated_content += file.read() + "\n"
            elif filename.endswith(".json"):
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    concatenated_content += json.dumps(data, indent=2) + "\n"
            elif filename.endswith(".docx"):
                doc = DocxDocument(filepath)
                concatenated_content += "\n".join([paragraph.text for paragraph in doc.paragraphs]) + "\n"
        return concatenated_content


    def answer_query(self, query):
        relevant_docs = self.knowledge_base.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        response_message = self.assistant.answer_query(query, context)
        return response_message

    def process_query(self, query):
        retrieved_docs = self.knowledge_base.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        return self.assistant.answer_query(query, context)