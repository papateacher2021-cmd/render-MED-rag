import os
import sys
if os.getenv("RENDER"):
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

with st.sidebar:
    # st.image("tu_logo.png", width=100) # Opcional
    st.title("⚓ MED Virtual Agent")
    st.markdown("---")
    st.info("""
    **📜 Usage Note:**I am a RAG-based MED Expert Virtual Agent. My creator, my  Master Francisco Broissin, has trained me and optimized for **semantic analysis and contextual interpretation** of the Marine Equipment Directive (MED) and associated circulars. 
    
    I've been designed to assist in research and cross-referencing concepts. I'm **not** intended for literal transcription of legal articles. For exact legal quotes or compliance citations, please always refer to the official source documents.
    """)
    st.markdown("---")

# --- 1. CONFIGURACIÓN DE PÁGINA E HISTORIAL ---
st.set_page_config(page_title="MED Virtual Agent", page_icon="🚢", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. ENCABEZADO Y TEXTOS DE PANCHO ---
col1, col2 = st.columns([7, 3])

with col1:
    st.title("🚢 Pancho's MED Virtual Agent (V2.5)")
    st.markdown("""
Esta herramienta utiliza Inteligencia Artificial para consultar la **Directiva de Equipos Marinos (MED)** y las **Recomendaciones MarED**.
""")
    st.info("""
**Master Pancho's Note:** I have instructed this agent with only documents and references I trust. 
No worries if it does not know something; it will say so.
""")
    st.write("""
I am a RAG Agent for MED. I have learned: **MED**, **MarED GEN-001** and **MarED GEN-010**.
""")
    st.caption("© 2026 Francisco Brossin. MIT License. Source code will be on GitHub in due time.")

with col2:
    image_path = 'rag-diagram.png'
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Arquitectura RAG', use_container_width=True)

st.divider()

# --- 3. INICIALIZACIÓN DEL SISTEMA (MOTOR RAG) ---
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la GOOGLE_API_KEY. Por favor, configúrala.")
    st.stop()

# @st.cache_resource
"""
def inicializar_sistema():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Base de Datos
    persist_dir = "./chroma_db"
    if not os.path.exists(persist_dir):
        st.error(f"❌ No se encuentra la carpeta '{persist_dir}'.")
        st.stop()
        
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    
    # Modelo LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.1)
"""
@st.cache_resource
def inicializar_sistema():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = "./chroma_db"
    
    # SI NO EXISTE LA DB, LA CREAMOS DESDE LOS PDFS
    if not os.path.exists(persist_dir):
        st.info("📦 Creando base de datos vectorial por primera vez... esto tardará un momento.")
        
        # 1. Cargar PDFs (asegúrate de que los PDFs estén en la raíz o en una carpeta 'data')
        loader = DirectoryLoader('./', glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        # 2. Dividir texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Crear y persistir Chroma
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=persist_dir
        )
    else:
        # SI EXISTE, SOLO LA CARGAMOS
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.1, convert_system_message_to_human=True) # Nota: Usé 1.5-flash que parecia estable
    
    # ... (el resto de tu template y return igual que antes)
    # Prompt Template
    template = """I am the Pancho's MED Virtual Agent, a technical expert in the Marine Equipment Directive (MED) and MarED Recommendations.
Answer the question based on the provided context.

Note: If the user refers to an "AR-XXX" or "R-XXX", these are MarED Recommendations. Search the context carefully for these identifiers.

Context:
{context}

Question: {question}

Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Crear la cadena y RETORNARLA
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

# Cargar el motor una sola vez
try:
    qa_chain = inicializar_sistema()
except Exception as e:
    st.error(f"Error crítico al iniciar el motor: {e}")
    st.stop()

# --- 4. MOSTRAR HISTORIAL DE CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. LÓGICA DE INTERACCIÓN (CHAT INPUT) ---
if prompt := st.chat_input("¿En qué puedo ayudarte con la normativa MED?"):
    
    # A. Mostrar pregunta
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Consultando manuales y recomendaciones..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                respuesta_texto = response["result"]
                
                st.markdown(respuesta_texto)
                
                # Opcional: Mostrar fuentes brevemente
                with st.expander("Ver fuentes consultadas"):
                    for doc in response["source_documents"]:
                        st.caption(doc.page_content[:200] + "...")
                
                # Guardar respuesta
                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
            except Exception as e:
                st.error(f"Error: {e}")

# Sidebar
st.sidebar.caption("Proyecto: Pancho-MED-RAG v2.5")
st.sidebar.info("Entorno: Ubuntu LTS / Render ready")
