# openai_client.py
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found!")

# Create a single client instance
openAI = OpenAI(api_key=api_key)