import openai
import logging
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_core.utils import is_request_type, is_intent_name

# Configurar logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clave de OpenAI
openai.api_key = "TU_CLAVE_OPENAI"  # Reemplaza con tu clave real

# Límite de mensajes en el historial
MAX_HISTORIAL = 5

def obtener_respuesta_openai(messages: list, temperature: float = 0.7) -> str:
    """
    Llama a la API de OpenAI usando ChatCompletion con el historial de mensajes.
    """
    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-4",  # Cambiar el modelo si es necesario
            messages=messages,
            max_tokens=1500,
            temperature=temperature
        )
        texto = respuesta.choices[0].message.content.strip()
        return texto if texto else "Lo siento, no tengo una respuesta para eso."
    except openai.error.OpenAIError as e:
        logger.error(f"Error en la API de OpenAI: {e}")
        return "No puedo procesar tu solicitud en este momento."
    except Exception as e:
        logger.error(f"Error general al comunicarse con OpenAI: {e}", exc_info=True)
        return "Hubo un error procesando tu solicitud."

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        session_attributes = handler_input.attributes_manager.session_attributes
        session_attributes["messages"] = []  # Inicializar historial de mensajes
        speech_text = "¡Hola! Estoy aquí para responder tus preguntas. ¿Qué te gustaría saber?"
        return handler_input.response_builder.speak(speech_text).ask(speech_text).response

class GeneralQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("GeneralQueryIntent")(handler_input)

    def handle(self, handler_input):
        try:
            # Obtener el slot de la pregunta
            slots = handler_input.request_envelope.request.intent.slots
            pregunta_usuario = slots["pregunta"].value if "pregunta" in slots and slots["pregunta"].value else None

            if not pregunta_usuario:
                return handler_input.response_builder.speak(
                    "No entendí tu pregunta. ¿Puedes repetirla?"
                ).ask("¿Puedes repetir tu pregunta?").response

            logger.info(f"Pregunta del usuario: {pregunta_usuario}")

            # Cargar historial de la sesión
            session_attributes = handler_input.attributes_manager.session_attributes
            if "messages" not in session_attributes:
                session_attributes["messages"] = []

            # Limitar historial a MAX_HISTORIAL mensajes
            if len(session_attributes["messages"]) >= MAX_HISTORIAL:
                session_attributes["messages"].pop(0)

            # Añadir pregunta al historial
            session_attributes["messages"].append({"role": "user", "content": pregunta_usuario})

            # Configurar temperatura basada en la longitud de la pregunta
            temperature = 0.7 if len(pregunta_usuario.split()) > 5 else 0.3

            # Obtener respuesta de OpenAI
            respuesta_openai = obtener_respuesta_openai(session_attributes["messages"], temperature)

            # Añadir respuesta al historial
            session_attributes["messages"].append({"role": "assistant", "content": respuesta_openai})

            return handler_input.response_builder.speak(respuesta_openai).ask(
                "¿Hay algo más que quieras saber?"
            ).response
        except Exception as e:
            logger.error(f"Error en GeneralQueryIntentHandler: {e}", exc_info=True)
            return handler_input.response_builder.speak(
                "Hubo un problema procesando tu solicitud. Por favor, inténtalo de nuevo."
            ).ask("¿Qué deseas preguntar?").response

class TopicQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("TopicQueryIntent")(handler_input)

    def handle(self, handler_input):
        try:
            slots = handler_input.request_envelope.request.intent.slots
            tema = slots["tema"].value if "tema" in slots and slots["tema"].value else None

            if not tema:
                return handler_input.response_builder.speak(
                    "No entendí el tema que mencionaste. ¿Puedes repetirlo?"
                ).ask("¿Puedes repetir el tema?").response

            logger.info(f"Tema solicitado: {tema}")

            # Respuesta predefinida simulada
            respuesta_openai = f"Aquí tienes información sobre {tema}. Estoy aprendiendo más sobre este tema."
            
            return handler_input.response_builder.speak(respuesta_openai).ask(
                "¿Te interesa algo más?"
            ).response
        except Exception as e:
            logger.error(f"Error en TopicQueryIntentHandler: {e}", exc_info=True)
            return handler_input.response_builder.speak(
                "Hubo un problema procesando tu solicitud. Por favor, inténtalo de nuevo."
            ).ask("¿Qué deseas preguntar?").response

class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        speech_text = ("Puedo ayudarte respondiendo preguntas sobre muchos temas. "
                       "Simplemente di: 'qué sabes sobre X' o 'dime algo sobre X'.")
        return handler_input.response_builder.speak(speech_text).ask("¿En qué más puedo ayudarte?").response

class FallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        speech_text = "Lo siento, no estoy segura de lo que quieres decir. Intenta preguntarme algo diferente."
        return handler_input.response_builder.speak(speech_text).ask("¿En qué más puedo ayudarte?").response

class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.CancelIntent")(handler_input) or is_intent_name("AMAZON.StopIntent")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.speak("Adiós").response

class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.response

class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        return handler_input.response_builder.speak(
            "Lo siento, ha ocurrido un error interno."
        ).ask("¿Puedes intentar nuevamente?").response

sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GeneralQueryIntentHandler())
sb.add_request_handler(TopicQueryIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
