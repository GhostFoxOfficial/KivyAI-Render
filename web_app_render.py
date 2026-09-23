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
    
    # CORRECTED HISTORY PARSING FOR GRADIO
    for chat_turn in history:
        if isinstance(chat_turn, dict):
            role = chat_turn.get("role")
            content = chat_turn.get("text", "")
            if role in ["user", "assistant"] and content:
                formatted_messages.append({"role": role, "content": content})
        elif isinstance(chat_turn, (list, tuple)) and len(chat_turn) == 2:
            if chat_turn[0]:
                formatted_messages.append({"role": "user", "content": chat_turn[0]})
            if chat_turn[1]:
                formatted_messages.append({"role": "assistant", "content": chat_turn[1]})
        
    formatted_messages.append({"role": "user", "content": message})

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
                        line_str = line_str[6:]
                    if line_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                partial_text += delta['content']
                                yield partial_text
                    except json.JSONDecodeError:
                        continue
        else:
            yield f"Error: Cloud AI server returned status code {response.status_code}. Please check your OpenRouter API key and balance."
            
    except requests.exceptions.Timeout:
        yield "Error: Connection timed out."
    except requests.exceptions.ConnectionError:
        yield "Error: Could not reach OpenRouter server."

demo = gr.ChatInterface(
    predict, 
    title="KivyAI", 
    description="Your fully AI."
)

if __name__ == "__main__":
    render_port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=render_port)
    
