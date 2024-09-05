import unittest
from llm_integration import analyze_code

class TestLLMIntegration(unittest.TestCase):
    def test_analyze_code(self):
        code_snippet = "def hello_world():\n    print('Hello, world!')"
        context = "No special context needed."
        result = analyze_code(code_snippet, context)
        self.assertIn("suggest", result.lower())

if __name__ == '__main__':
    unittest.main()