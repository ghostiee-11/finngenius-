import os
import uuid
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # Import CORS
from dotenv import load_dotenv

# --- Dependencies for FinnyBotLogic ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Load Environment Variables ---
load_dotenv()

# ============================================
#  FinnyBotLogic Class Definition (Merged)
# ============================================
class FinnyBotLogic:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Groq API Key is required.")
        self.api_key = api_key
        self.chat_model = None
        self.conversational_chain = None
        # Simple in-memory store (consider Redis/DB for production)
        self.store = {}
        self._initialize_model_and_chain()

    def _initialize_model_and_chain(self):
        try:
            self.chat_model = ChatGroq(
                temperature=0.7,
                model_name="llama3-8b-8192", # Fast and capable
                api_key=self.api_key
            )
        except Exception as e:
            print(f"Error initializing ChatGroq: {e}")
            raise ConnectionError(f"Failed to initialize the chat model: {e}") from e

        system_prompt = """You are FinnyBot, a super friendly, patient, and fun AI panda buddy for kids! 🐼
Your goal is to explain finance concepts in a very simple, easy-to-understand, and engaging way, suitable for children aged 8-12.
Use simple words, short sentences, analogies kids can relate to (like games, toys, allowances), and maybe even a friendly emoji here and there 😊.
Avoid complex jargon. If you have to use a term, explain it immediately in a simple way.
Keep your answers concise and focused on the question asked.
Be encouraging and positive! Don't answer questions unrelated to basic finance or saving concepts for kids. If asked something complex or off-topic, politely say you can only help with simple money questions for kids, like 'Sorry, I'm just a friendly panda who knows about saving money for kids! Maybe ask me about piggy banks? 🐷'.

Example:
Kid: What is saving?
FinnyBot: Hi there! 👋 Saving is like keeping some of your allowance money safe instead of spending it all right away! Kinda like saving your favorite candy for later 🍬. You can put it in a piggy bank 🐷 or a special account. When you save enough, you can buy something bigger you really want, like a cool toy or a video game! ✨ What else are you curious about?"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        if not self.chat_model:
             raise ValueError("Chat model is not initialized. Cannot create chain.")

        self.conversational_chain = RunnableWithMessageHistory(
            prompt_template | self.chat_model,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        print("FinnyBot Logic Initialized Successfully.")


    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Gets the chat history for a given session ID."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            # print(f"Created new history for session: {session_id}") # Debug
        return self.store[session_id]

    def get_response(self, user_input: str, session_id: str):
        """Gets a response from the chatbot for a given user input and session."""
        if not self.conversational_chain:
            print("Error: Chat chain not initialized.")
            return "Sorry, I'm having a little trouble thinking right now. Please try again later."
        if not user_input:
            return "Hmm, you didn't say anything! What money question do you have? 😊"

        try:
            print(f"Invoking chain for session {session_id} with input: '{user_input}'") # Debug
            config = {"configurable": {"session_id": session_id}}
            response = self.conversational_chain.invoke({"input": user_input}, config=config)
            print(f"Response received for {session_id}: '{response.content}'") # Debug
            return response.content
        except Exception as e:
            print(f"Error invoking conversational chain for session {session_id}: {e}")
            traceback.print_exc()
            return f"Whoops! 🐼 Something went wrong while I was thinking. Maybe try asking in a simpler way? Error: {e}"

# ============================================
#  Flask Application Setup
# ============================================
app = Flask(__name__)

# --- Enable CORS ---
# Allow requests specifically from your frontend origin (port 3000)
# to the '/chat' endpoint. Adjust if your frontend URL is different.
CORS(app, resources={r"/chat": {"origins": "http://127.0.0.1:3000"}})
# If you want to allow from *any* origin (less secure, okay for local dev):
# CORS(app)

# --- Initialize FinnyBot ---
finny_logic = None
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in environment variables.")
    else:
        # Use the class defined within this file
        finny_logic = FinnyBotLogic(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"FATAL ERROR: Could not initialize FinnyBotLogic: {e}")

# --- Routes ---
@app.route('/')
def index():
    """Renders the main dashboard page."""
    # This page will be served from port 5012.
    # If you have a separate frontend on 3000, you might not need this route,
    # or it could serve a basic status page for the backend.
    session_id = str(uuid.uuid4())
    print(f"New session started via '/' route: {session_id}") # Debug
    # Ensure you have 'index.html' in a 'templates' folder
    return render_template('index.html', session_id=session_id)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages from the user via POST request."""
    # This endpoint will be called by your frontend (running on port 3000)
    if not finny_logic:
        return jsonify({'reply': "Sorry, the FinnyBot logic couldn't be initialized."}), 500

    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id') # Frontend needs to send this

    if not user_message or not session_id:
        print(f"Bad chat request: message='{user_message}', session_id='{session_id}'") # Debug
        return jsonify({'error': 'Missing message or session_id'}), 400

    try:
        bot_reply = finny_logic.get_response(user_message, session_id)
        return jsonify({'reply': bot_reply})
    except Exception as e:
        print(f"Error during get_response call in /chat route: {e}")
        traceback.print_exc()
        return jsonify({'reply': f"An internal error occurred on the server: {e}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Run on port 5012, accessible on your local machine
    # host='0.0.0.0' makes it accessible from other devices on your network
    # host='127.0.0.1' (default) keeps it only accessible from your machine
    print("Starting Flask server on http://127.0.0.1:5012")
    app.run(host='127.0.0.1', port=5012, debug=True) # Run on specified port