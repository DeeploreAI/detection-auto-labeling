from google import genai
from openai import OpenAI


def configure_llm_client(source: str, api_key: str):
    """Configure LLM API, support OpenRouter and Google Gemini"""
    assert source in ["openrouter", "gemini"], "only support openrouter or google api."
    if source == "openrouter":
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        except Exception as e:
            print(f"OpenRouter API failed: {e}")
    elif source == "gemini":
        try:
            client = genai.Client(api_key=api_key)
            print("Configure Gemini API Successfully.")
        except Exception as e:
            print(f"Gemini API failed: {e}")
    return client