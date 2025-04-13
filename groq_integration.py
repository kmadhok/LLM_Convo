import os
import sys
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_llm_response(input_text, model="llama3-8b-8192"):
    """
    Get a response from Groq LLM.
    
    Args:
        input_text (str): The text to send to the LLM
        model (str): The Groq model to use
        
    Returns:
        str: The LLM response
    """
    print(f"input_text: {input_text}")
    # Get Groq API key from environment variable
    # api_key = os.environ.get("GROQ_API_KEY")
    # print(f"api_key: {api_key}")
    api_key = "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"
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set. Please set it with your Groq API key."
    
    try:
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond concisely to the user's input."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            model=model,
        )
        
        # Extract and return the response
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"Error: Failed to get response from Groq: {str(e)}"

def get_groq_response(input_text, model="llama3-8b-8192", history=None):
    #api_key = os.getenv("GROQ_API_KEY")
    api_key = "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"
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set. Please set it with your Groq API key."
    
    # Initialize history if not provided
    if history is None:
        history = []
    
    try:
        # Debug: Print incoming history
        print("\n" + "="*40 + " DEBUG HISTORY " + "="*40)
        print("Incoming history (", len(history), "messages):")
        print(json.dumps(history, indent=2))
        
        client = Groq(api_key=api_key)
        
        # Construct messages list
        messages = [{"role": "system", "content": "You are a helpful assistant. Respond concisely to the user's input."}]
        messages.extend(history)
        messages.append({"role": "user", "content": input_text})
        
        # Debug: Print full message payload
        print("\nFinal message payload:")
        print(json.dumps(messages, indent=2))
        print("="*100 + "\n")
        
        # API call
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"Error: Failed to get response from Groq: {str(e)}"
if __name__ == "__main__":
    # Check if input is provided via command line arguments
    if len(sys.argv) > 1:
        # The input text is the combined arguments (to handle spaces)
        #input_text = " ".join(sys.argv[1:])
        text="Hello, how are you?"
        response = get_llm_response(text)
        print(response)
    
    # Check if input is provided via stdin (pipe)
    elif not sys.stdin.isatty():
        # Read from stdin (for piping)
        input_text = sys.stdin.read().strip()
        if input_text:
            response = get_llm_response(input_text)
            print(response)
    
    else:
        print("Error: No input provided. Please provide text via command line arguments or pipe.")
        sys.exit(1)