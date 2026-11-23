import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("Iniciando la lectura de la biblioteca psicológica...")

# 1. Cargar todos los PDFs de la carpeta
loader = PyPDFDirectoryLoader("./base_conocimiento_psicologia")
documentos = loader.load()
print(f"-> Se cargaron {len(documentos)} páginas en total.")

# 2. Cortar los textos en fragmentos pequeños
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
fragmentos = text_splitter.split_documents(documentos)
print(f"-> Los documentos se dividieron en {len(fragmentos)} fragmentos exactos.")

# 3. Inicializar el modelo de lectura (Embeddings) de Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")
directorio_db = "./chroma_db"

print("-> Convirtiendo a vectores por lotes para no saturar la memoria. Esto tomará varios minutos...")

# 4. Crear la conexión a la base de datos vacía
vectorstore = Chroma(embedding_function=embeddings, persist_directory=directorio_db)

# 5. Procesar los fragmentos en lotes de 100
tamano_lote = 100
total_lotes = (len(fragmentos) // tamano_lote) + 1

for i in range(0, len(fragmentos), tamano_lote):
    lote = fragmentos[i : i + tamano_lote]
    vectorstore.add_documents(lote)
    num_lote_actual = (i // tamano_lote) + 1
    print(f"   [+] Procesado lote {num_lote_actual} de {total_lotes}...")

print("¡Éxito! La base de datos ha sido creada y guardada en la carpeta 'chroma_db'.")