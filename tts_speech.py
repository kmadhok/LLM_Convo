import boto3
import io
from pydub import AudioSegment
from pydub.playback import play
import os

def text_to_speech(text, voice_id="Ruth"):
    """
    Convert text to speech using AWS Polly and play it directly.
    
    Args:
        text (str): The text to convert to speech
        voice_id (str): The AWS Polly voice to use (default: Ruth)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Get AWS credentials from environment variables
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    
    # Check if credentials are available
    if not aws_access_key_id or not aws_secret_access_key:
        print("Error: AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        return False
    
    # Initialize Polly client
    try:
        polly_client = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        ).client('polly')
        
        # Request speech synthesis
        response = polly_client.synthesize_speech(
            Engine='neural',
            OutputFormat='mp3',
            SampleRate='24000',
            Text=text,
            TextType='text',
            VoiceId=voice_id
        )
        
        # Get the audio stream
        audio_stream = response['AudioStream']
        
        # Load the MP3 data directly from the stream
        sound = AudioSegment.from_file(io.BytesIO(audio_stream.read()), format="mp3")
        
        print(f"\n🔈 Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Play the audio
        play(sound)
        
        return True
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # If arguments are provided, use them as text
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        text_to_speech(input_text)
    
    # If no arguments but stdin has data (piping), use that
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
        if input_text:
            text_to_speech(input_text)
    
    # Otherwise, prompt for input
    else:
        text = input("Enter text to speak: HEELLLLOOOOOO")
        if text.strip():
            text_to_speech(text)