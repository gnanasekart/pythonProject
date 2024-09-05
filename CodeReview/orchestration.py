from pygments.lexers import guess_lexer

fixtures = {}

def detect_language(code_snippet):
    lexer = guess_lexer(code_snippet)
    return lexer.name

def should_review(language, files_changed):
    if language == "Python" and "security" in files_changed:
        return True
    return False

def create_prompt(code_snippet, context):
    instruction = "Focus on security vulnerabilities." if "security" in context else "General code review."
    prompt = f"{instruction}\n\n{context}\n\nReview this code:\n{code_snippet}"
    return prompt

def get_or_store_fixture(code_snippet, analysis):
    if code_snippet in fixtures:
        return fixtures[code_snippet]
    else:
        fixtures[code_snippet] = analysis
        return analysis
