from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Charger les variables d'environnement
load_dotenv()

# Créer un modèle ChatGemini
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # autres paramètres...
)

chat_history = []  # Utiliser une liste pour stocker les messages

# Définir un message système initial (optionnel)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # Ajouter le message système à l'historique

# Boucle de chat
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Ajouter le message utilisateur

    # Obtenir la réponse de l'IA en utilisant l'historique avec streaming
    print("AI: ", end="", flush=True)
    full_response = ""
    for chunk in model.stream(chat_history):
        chunk_content = chunk.content
        print(
            chunk_content, end="", flush=True
        )  # Afficher chaque morceau au fur et à mesure
        full_response += chunk_content

    chat_history.append(AIMessage(content=full_response))  # Ajouter le message de l'IA
    print("\n")  # Nouvelle ligne après la réponse

print("---- Historique des messages ----")
print(chat_history)
