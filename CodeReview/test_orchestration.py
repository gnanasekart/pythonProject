import unittest
from orchestration import detect_language, should_review, create_prompt, get_or_store_fixture

class TestOrchestration(unittest.TestCase):
    def test_detect_language(self):
        code_snippet = "def hello_world():\n    print('Hello, world!')"
        language = detect_language(code_snippet)
        self.assertEqual(language, "Python")

    def test_should_review(self):
        result = should_review("Python", ["security", "utils.py"])
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()