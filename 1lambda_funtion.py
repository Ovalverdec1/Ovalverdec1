import requests
import logging
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_core.utils import is_request_type, is_intent_name

# Configurar logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuración DeepSeek
DEEPSEEK_API_KEY = "tu-api-key-aqui"  # 🚨 Reemplazar con tu clave real
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"  # Modelo a usar

def obtener_respuesta_deepseek(messages: list) -> str:
    """Obtiene respuesta de la API de DeepSeek"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500
        }

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # Verificar errores HTTP

        respuesta_json = response.json()
        texto = respuesta_json['choices'][0]['message']['content'].strip()
        texto = " ".join(texto.split())  # Limpiar espacios
        
        return texto if texto else "Lo siento, no tengo una respuesta para eso."
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión: {str(e)}")
    except KeyError as e:
        logger.error(f"Error en estructura de respuesta: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
    
    return "Lo siento, no puedo responder en este momento."

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["messages"] = []
        
        speech = "¡Hola! Soy tu asistente con tecnología de DeepSeek. ¿En qué puedo ayudarte?"
        return (
            handler_input.response_builder
            .speak(speech)
            .ask("Puedes hacerme cualquier pregunta")
            .response
        )

class DeepSeekIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("OpenAIIntent")(handler_input)

    def handle(self, handler_input):
        slots = handler_input.request_envelope.request.intent.slots
        pregunta = slots["pregunta"].value if slots.get("pregunta") else None
        
        if not pregunta:
            return (
                handler_input.response_builder
                .speak("No escuché tu pregunta, ¿puedes repetirla?")
                .ask("Intenta formularlo de otra manera")
                .response
            )
        
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["messages"].append({"role": "user", "content": pregunta})
        
        respuesta = obtener_respuesta_deepseek(session_attr["messages"])
        
        # Limitar historial a últimos 5 mensajes para evitar sobrecarga
        session_attr["messages"].append({"role": "assistant", "content": respuesta})
        session_attr["messages"] = session_attr["messages"][-5:]  # Mantener contexto corto
        
        return (
            handler_input.response_builder
            .speak(respuesta)
            .ask("¿Necesitas algo más?")
            .response
        )

class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        return (
            handler_input.response_builder
            .speak("Pregúntame cualquier cosa. Por ejemplo: 'Qué es la inteligencia artificial'")
            .ask("¿Qué quieres saber?")
            .response
        )

class CancelStopHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (is_intent_name("AMAZON.CancelIntent")(handler_input) or
                is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        return handler_input.response_builder.speak("Hasta luego").set_should_end_session(True).response

class SessionEndedHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.response

class ErrorHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        return (
            handler_input.response_builder
            .speak("Ha ocurrido un error inesperado")
            .ask("Por favor intenta de nuevo")
            .response
        )

# Configurar Skill
sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(DeepSeekIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelStopHandler())
sb.add_request_handler(SessionEndedHandler())
sb.add_exception_handler(ErrorHandler())

lambda_handler = sb.lambda_handler()
