# https://python.langchain.com/docs/integrations/vectorstores/

from typing import List
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
import json
import redis


class RedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        """Chat Message History using Redis.

        Args:
            session_id: Identifiant unique pour la session de chat.
            redis_url: URL de connexion à Redis.
        """
        self.session_id = session_id
        self.redis = redis.from_url(redis_url)
        self.messages: List[BaseMessage] = []
        self._load_messages()

    def _load_messages(self) -> None:
        """Charge les messages depuis Redis."""
        self.redis.ltrim(self.session_id, 0, -1)  # Nettoyer les doublons
        messages = self.redis.lrange(self.session_id, 0, -1)
        if messages:
            # Convertir les messages JSON en objets BaseMessage
            self.messages = [
                HumanMessage(**json.loads(m))
                if json.loads(m)["type"] == "human"
                else AIMessage(**json.loads(m))
                for m in messages
            ]

    def add_message(self, message: BaseMessage) -> None:
        """Ajoute un message à l'historique."""
        self.messages.append(message)
        self.redis.rpush(self.session_id, message.model_dump_json())

    def clear(self) -> None:
        """Supprime tous les messages de l'historique."""
        self.messages = []
        self.redis.delete(self.session_id)


load_dotenv()

SESSION_ID = "user_session_new"

# Initialisation de Redis
print("Initialisation de Redis...")
chat_history = RedisChatMessageHistory(session_id=SESSION_ID)
print("Historique de chat initialisé.")
print("Historique actuel:", chat_history.messages)

# Initialisation du modèle Ollama
model = ChatOllama(
    model="llama3.2:3b-instruct-q8_0",
    temperature=0,
    # other params...
)

print("Commencez à discuter avec l'IA. Tapez 'exit' pour quitter.")
print("Tapez 'clear' pour effacer l'historique de la conversation.")
print("Tapez 'stream' pour activer le mode streaming.")

stream_mode = False

while True:
    user_input = input("Utilisateur: ")
    if user_input.lower() == "exit":
        break
    elif user_input.lower() == "clear":
        chat_history.clear()
        print("Historique de la conversation effacé.")
        continue
    elif user_input.lower() == "stream":
        stream_mode = not stream_mode
        print(f"Mode streaming {'activé' if stream_mode else 'désactivé'}")
        continue

    try:
        chat_history.add_message(HumanMessage(content=user_input))

        if stream_mode:
            print("IA: ", end="", flush=True)
            full_response = ""
            for chunk in model.stream(chat_history.messages):
                chunk_content = chunk.content
                print(chunk_content, end="", flush=True)
                full_response += chunk_content
            print()  # Nouvelle ligne après la fin du stream
            ai_message = AIMessage(content=full_response)
        else:
            ai_response = model.invoke(chat_history.messages)
            # Extraire uniquement le contenu de la réponse
            if isinstance(ai_response, BaseMessage):
                ai_content = ai_response.content
            elif isinstance(ai_response, dict) and "content" in ai_response:
                ai_content = ai_response["content"]
            else:
                ai_content = str(ai_response)
            ai_message = AIMessage(content=ai_content)
            print(f"IA: {ai_content}")

        chat_history.add_message(ai_message)
    except Exception as e:
        print(f"Erreur lors de la génération de la réponse: {e}")
