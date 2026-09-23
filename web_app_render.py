import json
import requests
import gradio as gr
import os

# OpenRouter configuration 
API_URL = "https://openrouter.ai/api/v1/chat/completions"
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
    
    # Universal history extraction loop
    if history:
        for chat_turn in history:
            if isinstance(chat_turn, dict):
                role = chat_turn.get("role")
                content = chat_turn.get("content") or chat_turn.get("text", "")
                if role in ["user", "assistant"] and str(content).strip():
                    formatted_messages.append({"role": role, "content": str(content).strip()})
            elif isinstance(chat_turn, (list, tuple)) and len(chat_turn) == 2:
                user_msg, assistant_msg = chat_turn
                if user_msg and str(user_msg).strip():
                    formatted_messages.append({"role": "user", "content": str(user_msg).strip()})
                if assistant_msg and str(assistant_msg).strip():
                    formatted_messages.append({"role": "assistant", "content": str(assistant_msg).strip()})
        
    if message and str(message).strip():
        formatted_messages.append({"role": "user", "content": str(message).strip()})
    else:
        yield "Error: Prompt cannot be empty."
        return

    payload = {
        "model": MODEL_NAME,
        "messages": formatted_messages,
        "stream": True
    }
    
    try:
        # Request stream with a clean headers definition
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60, stream=True)
        
        if response.status_code == 200:
            partial_text = ""
            # Hand over the streaming chunk assemblies to standard data generators
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8').strip()
                    
                    # Remove the SSE broadcast data tag safely
                    if line_str.startswith("data: "):
                        line_str = line_str[6:].strip()
                    
                    if line_str == "[DONE]" or not line_str:
                        continue
                    
                    try:
                        chunk = json.loads(line_str)
                        # Secure validation rules to safeguard against empty array indicators
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                partial_text += delta['content']
                                yield partial_text
                    except json.JSONDecodeError:
                        # If a partial chunk splits, ignore the error and wait for the rest of the string
                        continue
        else:
            yield f"Error: OpenRouter rejected the request with Status {response.status_code}. Raw response: {response.text}"
            
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
    
