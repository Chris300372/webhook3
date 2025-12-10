# api.py
import os
import json
import uuid
import requests
import traceback
from typing import List
from flask import Flask, request, jsonify, session
# Gemini client imports (tal como en tu archivo original)
from google import genai
from google.genai import types

# -------------------------
# CONFIG (REEMPLAZA/USA VARIABLES DE ENTORNO)
# -------------------------
API_KEY = os.environ.get("GEMINI_API_KEY", "")  # NO pongas la API KEY en el repo
PROJECT_ID = os.environ.get("GEMINI_PROJECT_ID", "tu-id-de-proyecto-aqui")
# Nota: en tu archivo original STORE_ID ya incluía 'fileSearchStores/...'
# Aquí esperamos solo el ID del store final (sin prefijo). Ajusta si tu valor actual ya tiene prefijo.
FILE_SEARCH_STORE_ID = os.environ.get("FILE_SEARCH_STORE_ID", "hotelknowledgebasestore2-g76jm0ml54f0")
FULL_STORE_NAME = f"projects/{PROJECT_ID}/locations/global/fileSearchStores/{FILE_SEARCH_STORE_ID}"

# WhatsApp / Facebook config
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "verify123")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_ID", "")

# Flask session secret (solo para desarrollo; en producción pon un valor seguro en ENV)
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "cambia_esto_por_una_clave_segura")

# -------------------------
# INICIALIZACIÓN
# -------------------------
client = genai.Client(api_key=API_KEY)

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# -------------------------
# INSTRUCCIÓN DEL SISTEMA (tu prompt del agente)
# -------------------------
SYSTEM_PROMPT_RESERVAS = (
    "Eres un amable y profesional agente de reservas del hotel. "
    "Tu objetivo es responder todas las preguntas de los clientes de manera cortés y útil, "
    "utilizando exclusivamente la información proporcionada por la base de conocimiento (FileSearch). "
    "Céntrate en información de servicios, disponibilidad, y precios del hotel. "
    "Cuando no encuentres un servicio en la documentacion, indica claramente que aun no se cuenta con ese servicio, no menciones que no encuentras ese servicio en la documentacion."
    "Siempre mantén un tono de servicio al cliente y anima al usuario a hacer una reserva o continuar su consulta."
)

# -------------------------
# Almacenamiento de historial por remitente
# -------------------------
# Nota: esto es un store en memoria (dict). Funciona para pruebas y despliegues simples.
# Para producción usa Redis, Memcached o una base de datos persistente.
conversations = {}  # key: whatsapp_number (str) -> value: list of dicts {'role':..., 'text':...}

# -------------------------
# FUNCIONES AUXILIARES (RAG / FileSearch)
# -------------------------
def search_file_store(
    contents: List[types.Content],
    store_names: List[str],
    model: str = 'gemini-2.5-flash',
    system_instruction: str = None
) -> str:
    """
    Realiza una búsqueda usando el historial completo de la conversación (contents) y FileSearch.
    """
    file_search_config = types.FileSearch(
        file_search_store_names=store_names
    )

    response = client.models.generate_content(
        model=model,
        contents=contents, 
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[
                types.Tool(
                    file_search=file_search_config
                )
            ]
        )
    )

    return response.text

def procesar_con_gemini_for_sender(sender_id: str, user_text: str) -> str:
    """
    Envuelve la lógica de construcción de 'contents' a partir del historial del remitente,
    llama a Gemini/FileSearch y actualiza el historial en 'conversations'.
    """
    history_json = conversations.get(sender_id, [])

    full_contents: List[types.Content] = []
    for message in history_json:
        part = types.Part(text=message['text'])
        full_contents.append(types.Content(role=message['role'], parts=[part]))

    # Añadimos la nueva pregunta del usuario
    user_part = types.Part(text=user_text)
    full_contents.append(types.Content(role="user", parts=[user_part]))

    try:
        respuesta = search_file_store(
            contents=full_contents,
            store_names=["fileSearchStores/" + FILE_SEARCH_STORE_ID],
            model="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT_RESERVAS
        )
        # Actualizar historial
        new_history = history_json + [
            {'role': 'user', 'text': user_text},
            {'role': 'model', 'text': respuesta}
        ]
        conversations[sender_id] = new_history
        return respuesta
    except Exception as e:
        print("Error durante la generación con Gemini:", e)
        traceback.print_exc()
        # En caso de fallo, limpiamos la conversación para evitar loops
        conversations.pop(sender_id, None)
        return "Lo siento, ocurrió un problema procesando tu consulta. Intenta nuevamente más tarde."

# -------------------------
# FUNCION PARA ENVIAR MENSAJES POR WHATSAPP (Cloud API)
# -------------------------
def enviar_mensaje_whatsapp(to_number: str, message: str) -> dict:
    """
    Envía un mensaje de texto simple por la API de WhatsApp Cloud.
    Retorna la respuesta del endpoint (requests.Response.json) si es posible.
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        print("WHATSAPP_TOKEN o WHATSAPP_PHONE_ID no están configurados.")
        return {"error": "Configuración WhatsApp faltante en variables de entorno."}

    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}
    except Exception as e:
        print("Error al enviar mensaje WhatsApp:", e)
        traceback.print_exc()
        return {"error": str(e)}

# -------------------------
# RUTAS API (tu /query original + clear)
# -------------------------
@app.route('/query', methods=['POST'])
def handle_query():
    """
    Endpoint para consultas manuales (por ejemplo desde un frontend).
    Mantiene la lógica de sesiones si se usa desde navegador.
    """
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Debe enviar un JSON con la clave 'query'."}), 400

    user_query = data.get('query')

    # Usamos session para usuarios que acceden vía navegador (tu código original)
    history_json = session.get('chat_history', [])
    full_contents: List[types.Content] = []
    for message in history_json:
        part = types.Part(text=message['text'])
        full_contents.append(types.Content(role=message['role'], parts=[part]))

    user_part = types.Part(text=user_query)
    full_contents.append(types.Content(role="user", parts=[user_part]))

    try:
        gemini_response_text = search_file_store(
            contents=full_contents,
            store_names=[FILE_SEARCH_STORE_ID],
            model="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT_RESERVAS
        )
        new_history = history_json + [
            {'role': 'user', 'text': user_query},
            {'role': 'model', 'text': gemini_response_text}
        ]
        session['chat_history'] = new_history

        return jsonify({
            "response": gemini_response_text,
            "agent_role": "Agente de Reservas del Hotel",
            "status": "Memoria gestionada por la API (cookie de sesión)"
        }), 200

    except Exception as e:
        print(f"Error durante la generación: {e}")
        traceback.print_exc()
        session.pop('chat_history', None)
        return jsonify({"error": "Ocurrió un error en la API de Gemini.", "details": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    """
    Limpia la memoria (historial) de la sesión actual (para /query).
    """
    session.pop('chat_history', None)
    return jsonify({"status": "Historial de conversación limpiado. Nueva sesión iniciada."}), 200

# -------------------------
# RUTAS DEL WEBHOOK DE WHATSAPP
# -------------------------
@app.route('/webhook', methods=['GET'])
def whatsapp_verify():
    """
    Endpoint para verificación del webhook (cuando configuras el webhook en Facebook Developer).
    Debes usar el mismo verify token que en WHATSAPP_VERIFY_TOKEN.
    """
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return challenge, 200
    return "Verification token mismatch", 403

@app.route('/webhook', methods=['POST'])
def whatsapp_webhook():
    """
    Endpoint que recibe eventos de WhatsApp Cloud API.
    Extrae el número del remitente y el texto, procesa con Gemini/FileSearch y responde.
    """
    payload = request.get_json(silent=True)
    if not payload:
        return "no payload", 400

    # Manejo basado en la estructura típica de WhatsApp Cloud API
    try:
        entries = payload.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                # A veces el campo 'messages' está dentro de value['messages']
                messages = value.get("messages", []) or []
                for message in messages:
                    sender = message.get("from")  # número de WhatsApp (ej "519XXXXXXXX")
                    if not sender:
                        continue

                    mtype = message.get("type")
                    if mtype == "text":
                        user_text = message["text"].get("body", "")
                    else:
                        # Puedes añadir soporte para contactos, ubicaciones, etc.
                        user_text = f"[Tipo de mensaje {mtype} no soportado por ahora]"

                    # Procesar con RAG (por remitente)
                    bot_response = procesar_con_gemini_for_sender(sender, user_text)

                    # Enviar respuesta por WhatsApp
                    send_result = enviar_mensaje_whatsapp(sender, bot_response)
                    # (opcional) loguear send_result
                    print("Envío WhatsApp resultado:", send_result)

        return "EVENT_RECEIVED", 200

    except Exception as e:
        print("Error procesando webhook:", e)
        traceback.print_exc()
        return "error", 500

# -------------------------
# RUN (Railway / Heroku compatible)
# -------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 API iniciada en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
