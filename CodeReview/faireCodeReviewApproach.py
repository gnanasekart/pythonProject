import openai

openai.api_key = "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"

response = openai.Completion.create(
    engine="gpt-3.5-turbo",
    prompt="Review this code:\n\n<code_snippet>",
    max_tokens=150
)

print(response.choices[0].text)
