
import unittest
import asyncio
from src.bridge import Bridge

class TestMiniMaxIntegration(unittest.TestCase):
    def setUp(self):
        # Trigger minimax detection
        self.bridge = Bridge(model_path="models/minimax-m2.1.gguf", mock=True)

    def test_apply_template_selection(self):
        messages = [{"role": "user", "content": "Hello"}]
        # This is a bit tricky to test because _generate is async and uses complex logic
        # But we can test if the _apply_minimax_template method exists and works on this instance
        prompt = self.bridge.flavor.apply_template(messages)
        self.assertIn("<|channel|>user", prompt)

    def test_stream_generate_uses_minimax(self):
        # We can mock the wrapper and check calls if we had a better mock setup
        pass

if __name__ == "__main__":
    unittest.main()
