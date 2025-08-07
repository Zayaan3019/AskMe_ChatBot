# backend/main.py

import os
import asyncio
import google.generativeai as genai
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()
app = FastAPI()

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url], # Use the variable here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the Gemini API client
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

# Pydantic model for a single message
class Message(BaseModel):
    role: str
    content: str

# Pydantic model for the entire conversation
class Conversation(BaseModel):
    messages: List[Message]

async def stream_generator(conversation: Conversation):
    """
    A generator function that streams the response from the Gemini API.
    """
    try:
        # Reformat messages for the Gemini API
        # Gemini uses 'parts' and expects the role to be 'user' or 'model'
        gemini_messages = [{"role": msg.role, "parts": [msg.content]} for msg in conversation.messages]

        # The last message is the new prompt, the rest is history.
        # However, the gemini-1.5-flash model takes the full history in generate_content.

        model = genai.GenerativeModel('gemini-1.5-flash')

        # Start streaming generation
        response_stream = model.generate_content(gemini_messages, stream=True)

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
                await asyncio.sleep(0.05) # Small delay for smooth streaming

    except Exception as e:
        print(f"Error during stream generation: {e}")
        yield "Sorry, an error occurred while generating the response."


@app.post("/api/chat")
async def chat_endpoint(conversation: Conversation):
    """
    This endpoint now handles a full conversation history and streams the response.
    """
    return StreamingResponse(stream_generator(conversation), media_type="text/plain")