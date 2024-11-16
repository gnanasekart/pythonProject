import openai

openai.api_key = "sk-"

response = openai.Completion.create(
    engine="gpt-3.5-turbo",
    prompt="Review this code:\n\n<code_snippet>",
    max_tokens=150
)

print(response.choices[0].text)
