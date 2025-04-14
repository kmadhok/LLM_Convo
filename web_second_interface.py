import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import threading
import time

# Updated imports to match app.py
from transcribe import no_wake_transcribe_from_microphone_simplified, transcribe_from_microphone_simplified
from groq_integration import get_llm_response, get_groq_response
from tts_speech import text_to_speech

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 👈 Add this here

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Flag to control the listening loop
is_listening = False
listen_thread = None

# Add history tracking from app.py
conversation_history = []
user_history = []
llm_history = []
history = []

@app.route('/')
def index():
    return render_template('index.html')

def background_listening():
    global is_listening, history, llm_history
    while is_listening:
        try:
            # Get user input from microphone - updated to use no_wake version
            transcript = no_wake_transcribe_from_microphone_simplified()
            
            if not transcript:
                continue
                
            # Send transcript to the web client
            socketio.emit('transcript', {'text': transcript})
            
            # Check if user wants to stop listening
            if "exit" in transcript.lower() or "quit" in transcript.lower() or "stop listening" in transcript.lower():
                is_listening = False
                socketio.emit('status', {'status': 'stopped'})
                break
                
            # Check for repetition like in app.py
            if llm_history and transcript.lower() == llm_history[-1].lower():
                response = "It seems you're repeating what I just said. Do you have a question about that?"
                socketio.emit('response', {'text': response})
            else:
                # Get response using Groq with history, matching app.py
                response = get_groq_response(
                    input_text=transcript,
                    model="llama3-8b-8192",
                    history=history
                )
                
                # Update histories
                llm_history.append(response)
                history.append({"role": "user", "content": transcript})
                history.append({"role": "assistant", "content": response})
                
                # Send response to the web client
                socketio.emit('response', {'text': response})
                
                # Speak the response
                text_to_speech(response)
                
        except Exception as e:
            print(f"Error in listening thread: {e}")
            socketio.emit('error', {'message': str(e)})

@socketio.on('start_listening')
def handle_start_listening():
    global is_listening, listen_thread
    
    if not is_listening:
        is_listening = True
        listen_thread = threading.Thread(target=background_listening)
        listen_thread.daemon = True
        listen_thread.start()
        emit('status', {'status': 'listening'})

@socketio.on('stop_listening')
def handle_stop_listening():
    global is_listening
    is_listening = False
    emit('status', {'status': 'stopped'})

@socketio.on('clear_history')
def handle_clear_history():
    global history, llm_history, user_history
    history = []
    llm_history = []
    user_history = []
    emit('status', {'status': 'history_cleared'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create HTML template - updated with a Clear History button
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        #conversation {
            height: 400px;
            border: 1px solid #ccc;
            padding: 10px;
            overflow-y: auto;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        .user-message {
            background-color: #dcf8c6;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 5px 0;
            max-width: 70%;
            align-self: flex-end;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #f1f0f0;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 5px 0;
            max-width: 70%;
        }
        .message-container {
            display: flex;
            flex-direction: column;
            margin-bottom: 10px;
        }
        .status {
            text-align: center;
            font-style: italic;
            color: #666;
            margin: 10px 0;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .controls {
            display: flex;
            justify-content: center;
        }
        .instructions {
            background-color: #e6f7ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 5px solid #1890ff;
        }
        #clearBtn {
            background-color: #f44336;
        }
        #clearBtn:hover {
            background-color: #d32f2f;
        }
    </style>
</head>
<body>
    <h1>Voice Assistant</h1>
    
    <div class="instructions">
        <h3>How to use:</h3>
        <p>1. Click "Start Listening" to begin</p>
        <p>2. Speak your question or command</p>
        <p>3. To end the session, say "exit", "quit", or "stop listening"</p>
        <p>4. Click "Clear History" to start a new conversation</p>
        <p><strong>Note:</strong> The assistant uses Llama 3 (8B) via Groq and retains conversation history.</p>
    </div>

    <div id="conversation"></div>
    
    <div class="controls">
        <button id="startBtn">Start Listening</button>
        <button id="stopBtn" disabled>Stop Listening</button>
        <button id="clearBtn">Clear History</button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io();
        const conversation = document.getElementById('conversation');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const clearBtn = document.getElementById('clearBtn');

        startBtn.addEventListener('click', () => {
            socket.emit('start_listening');
            addStatusMessage('Starting listening service...');
        });

        stopBtn.addEventListener('click', () => {
            socket.emit('stop_listening');
        });
        
        clearBtn.addEventListener('click', () => {
            socket.emit('clear_history');
            conversation.innerHTML = '';
            addStatusMessage('Conversation history cleared');
        });

        socket.on('status', (data) => {
            if (data.status === 'listening') {
                startBtn.disabled = true;
                stopBtn.disabled = false;
                addStatusMessage('System is listening. Speak your question...');
            } else if (data.status === 'stopped') {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                addStatusMessage('Listening stopped');
            } else if (data.status === 'history_cleared') {
                addStatusMessage('Conversation history has been cleared');
            }
        });

        socket.on('transcript', (data) => {
            addUserMessage(data.text);
        });

        socket.on('response', (data) => {
            addAssistantMessage(data.text);
        });

        socket.on('error', (data) => {
            addStatusMessage(`Error: ${data.message}`, true);
        });

        function addUserMessage(text) {
            const messageContainer = document.createElement('div');
            messageContainer.className = 'message-container';
            
            const message = document.createElement('div');
            message.className = 'user-message';
            message.textContent = text;
            
            messageContainer.appendChild(message);
            conversation.appendChild(messageContainer);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function addAssistantMessage(text) {
            const messageContainer = document.createElement('div');
            messageContainer.className = 'message-container';
            
            const message = document.createElement('div');
            message.className = 'assistant-message';
            message.textContent = text;
            
            messageContainer.appendChild(message);
            conversation.appendChild(messageContainer);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function addStatusMessage(text, isError = false) {
            const status = document.createElement('div');
            status.className = 'status';
            if (isError) {
                status.style.color = 'red';
            }
            status.textContent = text;
            conversation.appendChild(status);
            conversation.scrollTop = conversation.scrollHeight;
        }

        // Initial status message
        addStatusMessage('Welcome! Click "Start Listening" to begin.');
    </script>
</body>
</html>
        ''')
    
    print("Web interface starting at http://localhost:8080")
    # Make sure port matches Fly.io's expected port
    socketio.run(app, host='0.0.0.0', port=8080, debug=True, allow_unsafe_werkzeug=True)
