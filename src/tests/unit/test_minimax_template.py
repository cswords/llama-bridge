import unittest
import json
from src.bridge.flavors import MiniMaxFlavor

class TestMiniMaxTemplate(unittest.TestCase):
    def setUp(self):
        self.flavor = MiniMaxFlavor()

    def test_basic_conversation(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        prompt = self.flavor.apply_template(messages)
        expected = "<|channel|>system<|message|>You are helpful.<|channel|>user<|message|>Hello<|channel|>assistant<|message|>"
        self.assertEqual(prompt, expected)

    def test_multi_turn(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        prompt = self.flavor.apply_template(messages)
        expected = (
            "<|channel|>system<|message|>You are helpful."
            "<|channel|>user<|message|>Hello"
            "<|channel|>assistant<|message|>Hi there!"
            "<|channel|>user<|message|>How are you?"
            "<|channel|>assistant<|message|>"
        )
        self.assertEqual(prompt, expected)

    def test_with_tools(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        tools = [
            {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}
        ]
        prompt = self.flavor.apply_template(messages, tools)
        
        # Check if tools are in the system message
        self.assertIn("<|channel|>system<|message|>You are helpful.", prompt)
        self.assertIn("get_weather", prompt)
        self.assertIn("Get weather", prompt)
        self.assertTrue(prompt.endswith("<|channel|>assistant<|message|>"))

    def test_no_system_message_with_tools(self):
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        tools = [{"name": "tool1"}]
        prompt = self.flavor.apply_template(messages, tools)
        
        self.assertIn("<|channel|>system<|message|>", prompt)
        self.assertIn("tool1", prompt)
        self.assertIn("<|channel|>user<|message|>Hello", prompt)

if __name__ == "__main__":
    unittest.main()
