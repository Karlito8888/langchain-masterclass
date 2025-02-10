# Chat Model Documents: https://python.langchain.com/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/docs/integrations/chat/ollama/

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
# Load environment variables from .env
load_dotenv()

# Create a ChatOllama model
model = ChatOllama(
    model="llama3.2:3b-instruct-q8_0",
    temperature=0,
    # other params...
)

# Invoke the model with a message
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
