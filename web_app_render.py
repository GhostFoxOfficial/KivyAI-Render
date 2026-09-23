import json
import requests
import gradio as gr
import os

# OpenRouter configuration 
API_URL = "https://openrouter.ai"
MODEL_NAME = "qwen/qwen-2.5-3b-instruct:free"

def predict(message, history):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        yield "Error: OPENROUTER_API_KEY environment variable is not set on Render."
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://render.com", 
        "X-Title": "KivyAI Web"
    }
    
    formatted_messages = [
        {
            "role": "system",
            "content": "Your name is KivyAI. You are cognitive, sentient, omniscient, friendly, and kind. You can speak in any language. You can code/program. You are created by Ghost Fox (individual)."
        }
    ]
    
    # REFIXED: Strict text extraction to guarantee no empty payloads
    if history:
        for chat_turn in history:
            if isinstance(chat_turn, dict):
                role = chat_turn.get("role")
                content = chat_turn.get("content") or chat_turn.get("text", "")
                if role in ["user", "assistant"] and str(content).strip():
                    formatted_messages.append({"role": role, "content": str(content).strip()})
            
            elif isinstance(chat_turn, (list, tuple)) and len(chat_turn) == 2:
                user_msg, assistant_msg = chat_turn
                # Only append if the strings are not completely empty space
                if user_msg and str(user_msg).strip():
                    formatted_messages.append({"role": "user", "content": str(user_msg).strip()})
                if assistant_msg and str(assistant_msg).strip():
                    formatted_messages.append({"role": "assistant", "content": str(assistant_msg).strip()})
        
    # Safely append user message
    if message and str(message).strip():
        formatted_messages.append({"role": "user", "content": str(message).strip()})
    else:
        yield "Error: Cannot send an empty prompt message."
        return

    payload = {
        "model": MODEL_NAME,
        "messages": formatted_messages,
        "stream": True
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60, stream=True)
        
        if response.status_code == 200:
            partial_text = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith("data: "):
                        line_str = line_str[len("data: "):].strip()
                        
                    if line_str == "[DONE]" or not line_str:
                        continue
                        
                    try:
                        chunk = json.loads(line_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                partial_text += delta['content']
                                yield partial_text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        else:
            # Let's read the error message directly from OpenRouter so you know exactly why it fails
            error_details = response.text
            yield f"Error: OpenRouter rejected the request (Status Code {response.status_code}). Details: {error_details}"
            
    except requests.exceptions.Timeout:
        yield "Error: Cloud connection timed out."
    except requests.exceptions.ConnectionError:
        yield "Error: Unable to connect to OpenRouter server."

demo = gr.ChatInterface(
    predict, 
    title="KivyAI", 
    description="Your fully AI."
)

if __name__ == "__main__":
    render_port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=render_port)
    
