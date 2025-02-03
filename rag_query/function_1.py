import azure.functions as func
import logging
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
from langchain.schema import Document
from openai import OpenAI
from docx import Document as DocxDocument
import os
import json
import tempfile
import pickle
from azure.storage.blob import BlobServiceClient

os.environ['FAISS_NO_GPU'] = '1'

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

    def build_retriever(self, connection_string, container_name):
        store = FAISS.from_documents(self.documents, self.embeddings)
        self._save_to_blob(store, connection_string, container_name)
        self.retriever = store.as_retriever()
        # if save:
        #     store.save_local("faiss_index")
        # self.retriever =store.as_retriever()

    @staticmethod
    def _save_to_blob(store, connection_string, container_name):
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Save the index to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as temp_index:
            faiss.write_index(store.index, temp_index.name)
            # Upload to blob storage
            with open(temp_index.name, 'rb') as data:
                container_client.upload_blob(name="index.faiss", data=data, overwrite=True)
        os.unlink(temp_index.name)

        # Save the docstore
        with tempfile.NamedTemporaryFile(delete=False) as temp_docstore:
            pickle.dump(store.docstore, temp_docstore)
            temp_docstore.flush()
            with open(temp_docstore.name, 'rb') as data:
                container_client.upload_blob(name="docstore.pkl", data=data, overwrite=True)
        os.unlink(temp_docstore.name)

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
        self.connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        self.container_name = os.environ["AZURE_STORAGE_CONTAINER_NAME"]
        try:
            # Try to load from blob storage
            self.knowledge_base = KnowledgeBase('')
            self.knowledge_base.retriever = self._load_from_blob().as_retriever()
        except Exception as e:
            logging.info(f"Could not load from blob storage: {str(e)}. Building new index.")
            documents = self.load_files_contents('data')
            self.knowledge_base = KnowledgeBase(documents)
            self.knowledge_base.build_retriever(self.connection_string, self.container_name)

        self.assistant = VirtualAssistant(
            retriever=self.knowledge_base.retriever,
            general=True,
            llm_model=llm_model
        )
        # if os.path.exists("faiss_index"):
        #     self.knowledge_base = KnowledgeBase('')
        #     self.knowledge_base.retriever = self.load_local_cpu_index("faiss_index", self.knowledge_base.embeddings).as_retriever()
        # else:
        #     documents = self.load_files_contents('data')
        #     self.knowledge_base = KnowledgeBase(documents)
        #     self.knowledge_base.build_retriever()

        # self.assistant = VirtualAssistant(retriever=self.knowledge_base.retriever,general=True,llm_model=llm_model)

    def _load_from_blob(self):
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(self.container_name)

        # Download index
        with tempfile.NamedTemporaryFile(delete=False) as temp_index:
            index_blob_client = container_client.get_blob_client("index.faiss")
            index_data = index_blob_client.download_blob()
            temp_index.write(index_data.readall())
            temp_index.flush()
            
            # Download docstore
            docstore_blob_client = container_client.get_blob_client("docstore.pkl")
            docstore_data = docstore_blob_client.download_blob()
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_docstore:
                temp_docstore.write(docstore_data.readall())
                temp_docstore.flush()
                
                # Load both files
                index = faiss.read_index(temp_index.name)
                with open(temp_docstore.name, 'rb') as f:
                    docstore = pickle.load(f)

        # Cleanup temporary files
        os.unlink(temp_index.name)
        os.unlink(temp_docstore.name)

        # Create FAISS instance
        faiss_instance = FAISS(self.knowledge_base.embeddings.embed_query, index, docstore)
        return faiss_instance
    
    @staticmethod
    def load_local_cpu_index(folder_path, embeddings):
        index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        faiss.normalize_L2(index)
        return FAISS.load_local(folder_path, embeddings, index=index)

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
        print(f"Retrieved documents: {retrieved_docs}")
        if not retrieved_docs:
            return "Aucun document pertinent trouvé."
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        return self.assistant.answer_query(query, context)