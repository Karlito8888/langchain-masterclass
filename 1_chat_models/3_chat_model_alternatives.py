from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print(f"Answer from Google: {result.content}")


# ---- Ollama Chat Model Example ----

model = ChatOllama(
    model="llama3.2:3b-instruct-q8_0",
    temperature=0.4,
    # other params...
)

result = model.invoke(messages)
print(f"Answer from Ollama: {result.content}")


# ---- LangChain OpenAI Chat Model Example ----

# Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o")

# # Invoke the model with messages
# result = model.invoke(messages)
# print(f"Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----

# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
# model = ChatAnthropic(model="claude-3-opus-20240229")

# result = model.invoke(messages)
# print(f"Answer from Anthropic: {result.content}")


