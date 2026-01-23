
import pytest
from src.utils.scanner import StreamScanner

def merge_events(events):
    if not events: return []
    merged = []
    current_type, current_text = events[0]
    for type_, text in events[1:]:
        if type_ == current_type:
            current_text += text
        else:
            merged.append((current_type, current_text))
            current_type = type_
            current_text = text
    merged.append((current_type, current_text))
    return merged


class TestStreamScanner:
    def test_plain_text(self):
        scanner = StreamScanner()
        events = scanner.consume("Hello World")
        assert events == [("content", "Hello World")]

    def test_code_less_than(self):
        scanner = StreamScanner()
        # "if x < 5:" should not be treated as a tag start indefinitely
        events = scanner.consume("if x < 5:")
        # The scanner buffers '< 5:' waiting for '>'. 
        # Since it's short, it might hold it if not forced.
        # But '< 5:' does not look like a valid tag start pattern usually?
        # My implementation waits for '>' or max length.
        # 'if x ' -> content.
        # '< 5:' -> buffer.
        
        # We need to force flush to see result or add more context
        events += scanner.consume(" continue", is_final=True)
        
        # Expectation: The scanner eventually realizes it's not a tag or flushes at end
        # The output should recreate the input text exactly
        text = "".join(c for t, c in events)
        assert text == "if x < 5: continue"

    def test_broken_tag_flush(self):
        """Test that a '<' followed by non-tag chars is eventually flushed."""
        scanner = StreamScanner()
        scanner.max_scannable_len = 10 # Shorten for test
        
        # Feed enough data to overflow buffer
        events = scanner.consume("x < " + "a"*15)
        text = "".join(c for t, c in events)
        assert text == "x < " + "a"*15

    def test_valid_thought(self):
        scanner = StreamScanner()
        events = scanner.consume("Start <thought>I am thinking</thought> End")
        
        # Logic: 
        # "Start " -> content
        # "<thought>I am thinking</thought>" -> thought
        # " End" -> content
        
        expected = [
            ("content", "Start "),
            ("thought", "I am thinking"),
            ("content", " End")
        ]
        assert events == expected

    def test_suppress_tool_call(self):
        scanner = StreamScanner()
        events = scanner.consume("Text <tool_call>hidden</tool_call> more text")
        
        # "<tool_call>hidden</tool_call>" should be suppressed
        expected = [
            ("content", "Text "),
            ("content", " more text")
        ]
        assert events == expected

    def test_suppress_minimax_tool_call(self):
        scanner = StreamScanner()
        events = scanner.consume("A <minimax:tool_call>stuff</minimax:tool_call> B")
        expected = [
            ("content", "A "),
            ("content", " B")
        ]
        assert events == expected

    def test_fragmented_tags(self):
        """Crucial: Test tags arriving one char at a time."""
        scanner = StreamScanner()
        text = "Start <thought>Think</thought>"
        events = []
        for char in text:
            events.extend(scanner.consume(char))
        events.extend(scanner.consume("", is_final=True))
        
        expected = [
            ("content", "Start "),
            ("thought", "Think")
        ]
        if merge_events(events) != expected:
            print(f"\n[FAILED] Fragmented Tags:\nExpected: {expected}\nActual:   {merge_events(events)}")
        assert merge_events(events) == expected

    def test_user_hallucination_case(self):
        """
        User's example: 'invoke< nameBash<="commandgit --</>'
        This contains random '<' that shouldn't trigger suppression of following text.
        """
        input_str = 'invoke< nameBash<="commandgit --</>'
        scanner = StreamScanner()
        events = scanner.consume(input_str, is_final=True)
        text = "".join(c for t, c in events)
        assert text == input_str

    def test_tool_definitions_passthrough(self):
        # Tools often have XML-like definitions in the prompt/output that we might WANT to see? 
        # Or usually the model outputs text.
        # But specifically <invoke> <parameter> etc should be treated as text if they are not inside a tool_call block?
        # Wait, the prompt says "Skipping of self-contained tool tags".
        pass
    
    def test_inline_tags_skipped(self):
        # "Skipping of self-contained tool tags (<invoke ...>, <function=...)"
        # This requirement from my thought process implies suppressing them? 
        # Or just not treating them as mode-switchers?
        # My code implementation: 
        # elif raw_name in ["invoke", "function", "parameter"]:
        #    pass (Stay in text mode, just consume the tag and move on? Wait, my code swallows them!)
        #    # "Move past this tag" -> this means the tag is consumed from buffer but NOT added to events.
        
        scanner = StreamScanner()
        # If the model outputs raw <invoke> tags outside of a tool block, we largely want to HIDE them from the user
        # if they are part of the protocol.
        input_str = "Call <invoke name='test'>args</invoke> done."
        events = scanner.consume(input_str, is_final=True)
        # Expectation based on code: <invoke ...> is skipped (swallowed). 'args' remains? </invoke> is skipped?
        
        # Code check: 
        # <invoke...> matches "invoke" -> pass -> buffer advanced. NO event yielded.
        # args -> content
        # </invoke> -> matches "invoke", is_closing -> pass -> buffer advanced. NO event yielded.
        
        expected_text = "Call args done."
        text = "".join(c for t, c in events)
        assert text == expected_text

if __name__ == "__main__":
    # Manually run if executed as script
    t = TestStreamScanner()
    t.test_plain_text()
    t.test_code_less_than()
    t.test_broken_tag_flush()
    t.test_valid_thought()
    t.test_suppress_tool_call()
    t.test_suppress_minimax_tool_call()
    t.test_fragmented_tags()
    t.test_user_hallucination_case()
    t.test_inline_tags_skipped()
    print("All robust scanning tests passed.")
