import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def analyze_code(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text
