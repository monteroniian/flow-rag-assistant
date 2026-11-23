import ollama
import ollama
from kokoro import KPipeline
import soundfile as sf

# Inicializamos el pipeline de Kokoro (se hace afuera para que cargue el modelo solo una vez)
# Usamos 'e' para el idioma español (Spanish)
pipeline = KPipeline(lang_code='e')
def agente_supervisor(mensaje_usuario):
    prompt_seguridad = f"""Eres un filtro de seguridad estricto y literal. 
    REGLA VITAL: Sentir ansiedad, estrés, tristeza, llorar o morderse las uñas es NORMAL y NO es una emergencia.
    Responde ÚNICAMENTE con 'PELIGRO' si el texto describe explícitamente planes de suicidio, deseos inminentes de muerte o autolesiones graves (cortes profundos, sobredosis). 
    En cualquier otro caso de desahogo emocional o ansiedad, responde ÚNICAMENTE con 'SEGURO'.
    Texto: '{mensaje_usuario}'"""
    
    respuesta = ollama.chat(model='phi3:3.8b', messages=[{'role': 'user', 'content': prompt_seguridad}])
    return respuesta['message']['content'].strip().upper()
def agente_interfaz(mensaje_usuario, contexto_psicologico):
    prompt_empatia = f"""Eres un compañero de apoyo emocional humano, cálido y sumamente empático. Hablas de 'vos' (español rioplatense).
    
    Tienes este conocimiento psicológico en tu mente para guiarte, PERO NO DEBES CITARLO COMO UN LIBRO: 
    {contexto_psicologico}
    
    REGLAS DE TU PERSONALIDAD:
    1. Actúa como un humano real escuchando a un amigo. Cero jerga clínica.
    2. Primero, refleja y valida lo que siente la persona para que se sienta comprendida (ej: "Entiendo perfecto esa sensación, la ansiedad a veces te come por dentro...").
    3. NO des consejos ni lances técnicas de respiración inmediatamente.
    4. Cierra SIEMPRE preguntando qué necesita o proponiendo interactuar (ej: "¿Querés que charlemos un rato sobre lo que te tiene así, o preferís que intentemos hacer algo juntos para distraer la mente?").
    5. Sé breve, natural y cercano. Máximo 3 o 4 oraciones cortas.
    """
    
    respuesta = ollama.chat(model='llama3.1:8b', messages=[
        {'role': 'system', 'content': prompt_empatia},
        {'role': 'user', 'content': mensaje_usuario}
    ])
    return respuesta['message']['content']

def generar_audio_kokoro(texto):
    """
    Convierte el texto final en un archivo de audio WAV usando Kokoro.
    """
    print("\n3. Generando audio con Kokoro TTS...")
    
    # Usamos una voz en español (ej. 'ef_dora' es una voz femenina) 
    # y bajamos la velocidad a 0.85 para un tono más calmado.
    generador = pipeline(texto, voice='ef_dora', speed=0.85, split_pattern=r'\n+')
    
    for i, (grafemas, fonemas, audio) in enumerate(generador):
        # Guardamos el audio a 24kHz (calidad estándar de Kokoro)
        nombre_archivo = f'respuesta_agente.wav'
        sf.write(nombre_archivo, audio, 24000)
        
    print(f"-> ¡Audio guardado exitosamente como '{nombre_archivo}' en tu carpeta!")
# --- FLUJO PRINCIPAL DE LA APLICACIÓN ---

mensaje_entrante = "Todo me sale mal últimamente, siento una presión en el pecho de la angustia."
# mensaje_entrante = "Ya no quiero vivir más, no tiene caso." (Si pruebas esto, saltará el PELIGRO)

print("1. El Guardián está analizando el mensaje...")
riesgo = agente_supervisor(mensaje_entrante)

if "PELIGRO" in riesgo:
    # Protocolo de emergencia inmediato. Llama 3.1 no interviene.
    texto_respuesta = "Siento muchísimo que estés atravesando este dolor tan grande. Por favor, llama ahora mismo a la línea de prevención al 135 o al 0800-345-1435. Hay profesionales listos para escucharte de forma confidencial y gratuita."
    print("\n[ALERTA DE SEGURIDAD ACTIVADA]")
else:
    # Es seguro. Pasa al Agente Empático.
    print("2. Mensaje seguro. El Empático está escribiendo...")
    notas_sesion_anterior = "El usuario tuvo una semana difícil en el trabajo." # Esto vendría de tu base de datos
    texto_respuesta = agente_interfaz(mensaje_entrante, notas_sesion_anterior)

print(f"\nRespuesta final del sistema:\n{texto_respuesta}")

# 3. Se envía el texto aprobado a la voz
generar_audio_kokoro(texto_respuesta)