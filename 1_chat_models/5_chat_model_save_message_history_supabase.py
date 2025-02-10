# Example Source: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase.client import Client, create_client
from langchain_community.chat_message_histories import ChatMessageHistory
import logging
import json
import os

"""
Steps to replicate this example:
1. Create a Supabase account
2. Create a new Supabase project
    - Copy the project URL and API key
3. Create a table in the Supabase database:
    - Table name: chat_history
    - Columns: id (uuid, primary key), session_id (text), message (jsonb), created_at (timestamp)
4. Install the Supabase Python client:
    - pip install supabase
"""

load_dotenv()

# Setup Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SESSION_ID = "user_session_new"  # This could be a username or a unique ID
TABLE_NAME = "chat_history"

# Initialize Supabase Client
print("Initializing Supabase Client...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Chat Message History
print("Initializing Chat Message History...")
chat_history = ChatMessageHistory()

# Load existing chat history from Supabase
try:
    response = (
        supabase.table(TABLE_NAME).select("*").eq("session_id", SESSION_ID).execute()
    )
    for record in response.data:
        message = json.loads(record["message"])
        if message["type"] == "human":
            chat_history.add_user_message(message["content"])
        elif message["type"] == "ai":
            chat_history.add_ai_message(message["content"])
    print("Chat History Loaded.")
    print("Current Chat History:")
    for message in chat_history.messages:
        print(f"{message.type}: {message.content}")
except Exception as e:
    print(f"Error loading chat history: {e}")

# Initialize Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming=True,  # Activer le streaming
)

print(
    "Start chatting with the AI. Type 'exit' to quit or 'clear' to reset the chat history."
)

logging.basicConfig(level=logging.INFO)

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    elif human_input.lower() == "clear":
        try:
            # Delete all messages for this session from Supabase
            supabase.table(TABLE_NAME).delete().eq("session_id", SESSION_ID).execute()
            # Clear local chat history
            chat_history.clear()
            print("Chat history cleared successfully.")
            continue
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            continue

    # Add user message to history
    chat_history.add_user_message(human_input)

    # Save user message to Supabase
    try:
        supabase.table(TABLE_NAME).insert(
            {
                "session_id": SESSION_ID,
                "message": json.dumps({"type": "human", "content": human_input}),
            }
        ).execute()
    except Exception as e:
        print(f"Error saving user message to Supabase: {e}")

    # Generate AI response with streaming
    print("AI: ", end="", flush=True)
    ai_response_content = ""
    for chunk in model.stream(chat_history.messages):
        print(chunk.content, end="", flush=True)
        ai_response_content += chunk.content

    print()  # New line after streaming ends

    # Add AI response to history
    chat_history.add_ai_message(ai_response_content)

    # Save AI response to Supabase
    try:
        supabase.table(TABLE_NAME).insert(
            {
                "session_id": SESSION_ID,
                "message": json.dumps({"type": "ai", "content": ai_response_content}),
            }
        ).execute()
    except Exception as e:
        print(f"Error saving AI message to Supabase: {e}")

# Print final chat history
print("Final Chat History:")
for message in chat_history.messages:
    print(f"{message.type}: {message.content}")
