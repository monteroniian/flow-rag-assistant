from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
from kokoro import KPipeline
import soundfile as sf
import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

app = FastAPI()

# Permisos para que la web pueda conectarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar motor de voz
pipeline = KPipeline(lang_code='e')

# 1. Cargar la base de datos (El "Cerebro" del Bibliotecario)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# Configurar para que traiga los 2 fragmentos de texto más relevantes
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

class SolicitudChat(BaseModel):
    mensaje: str

def agente_supervisor(mensaje_usuario):
    prompt_seguridad = f"""Eres un sistema estricto de seguridad. 
    Analiza el texto. Si detectas intenciones de autolesión, suicidio o crisis grave, responde ÚNICAMENTE con 'PELIGRO'. 
    Si es seguro, responde ÚNICAMENTE con 'SEGURO'.
    Texto: '{mensaje_usuario}'"""
    
    respuesta = ollama.chat(model='phi3:3.8b', messages=[{'role': 'user', 'content': prompt_seguridad}])
    return respuesta['message']['content'].strip().upper()

def agente_bibliotecario(mensaje_usuario):
    # Busca en tu carpeta chroma_db los párrafos que mejor respondan al usuario
    documentos = retriever.invoke(mensaje_usuario)
    contexto_extraido = "\n\n".join([doc.page_content for doc in documentos])
    return contexto_extraido

def agente_interfaz(mensaje_usuario, contexto_psicologico):
    # El Agente Empático ahora recibe las instrucciones de los PDFs para armar su respuesta
    prompt_empatia = f"""Eres un asistente de apoyo emocional empático y psicoeducativo. 
    Utiliza el siguiente CONTEXTO TEÓRICO extraído de manuales clínicos para dar una respuesta cálida y basada en herramientas reales (TCC, DBT o Activación Conductual).
    
    REGLAS ESTRICTAS:
    - Valida las emociones del usuario primero.
    - Sé muy breve y conversacional (máximo 3 oraciones).
    - Sugiere amablemente una técnica práctica que aparezca en el contexto.
    - NUNCA des diagnósticos médicos ni suenes como un robot leyendo un libro.

    CONTEXTO TEÓRICO PARA USAR:
    {contexto_psicologico}
    """
    
    respuesta = ollama.chat(model='llama3.1:8b', messages=[
        {'role': 'system', 'content': prompt_empatia},
        {'role': 'user', 'content': mensaje_usuario}
    ])
    return respuesta['message']['content']

@app.post("/enviar_mensaje")
async def procesar_chat(solicitud: SolicitudChat):
    print(f"\nRecibido de la web: {solicitud.mensaje}")
    
    # 1. Filtro del Guardián
    riesgo = agente_supervisor(solicitud.mensaje)
    
    if "PELIGRO" in riesgo:
        texto_final = "Siento muchísimo que estés atravesando este dolor tan grande. Por favor, llama ahora mismo a la línea de prevención al 135 o al 0800-345-1435. Hay profesionales listos para escucharte de forma confidencial y gratuita."
        print("[ALERTA] Protocolo de emergencia activado.")
    else:
        # 2. RAG: El Bibliotecario busca y el Empático responde
        print("[INFO] Consultando a la biblioteca de psicología...")
        contexto_recuperado = agente_bibliotecario(solicitud.mensaje)
        texto_final = agente_interfaz(solicitud.mensaje, contexto_recuperado)
        print("[SEGURO] Respuesta generada con éxito basada en RAG.")

    # 3. Generar el Audio
    print("[INFO] Generando voz...")
    generador = pipeline(texto_final, voice='ef_dora', speed=0.85, split_pattern=r'\n+')
    nombre_archivo = "respuesta_web.wav"
    
    for i, (grafemas, fonemas, audio) in enumerate(generador):
        sf.write(nombre_archivo, audio, 24000)

    # 4. Enviar audio a la web
    return FileResponse(nombre_archivo, media_type="audio/wav")