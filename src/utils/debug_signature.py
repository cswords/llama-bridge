import os
import sys
from pathlib import Path

# Setup paths to find compiled llama_chat binding in parent directory (src/)
_utils_dir = Path(__file__).parent.absolute()
_src_dir = _utils_dir.parent

if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

if sys.platform == "darwin":
    lib_path = _src_dir / "lib"
    if "DYLD_LIBRARY_PATH" not in os.environ:
        os.environ["DYLD_LIBRARY_PATH"] = str(lib_path)
    else:
        path = str(lib_path)
        if path not in os.environ["DYLD_LIBRARY_PATH"]:
            os.environ["DYLD_LIBRARY_PATH"] = f"{path}:{os.environ['DYLD_LIBRARY_PATH']}"

import llama_chat

model_path = "models/unsloth/MiMo-V2-Flash-GGUF/UD-Q6_K_XL/MiMo-V2-Flash-UD-Q6_K_XL-00001-of-00006.gguf"

print("Docstring:", llama_chat.LlamaChatWrapper.__init__.__doc__)

try:
    # Try with new signature: path, n_ctx, n_batch, n_ubatch, n_threads, flash_attn
    # Just checking signature match, not full load (passing 0 ctx to fail fast or just minimal)
    # We can pass 0 for ctx/batch to use defaults, but we want to test arg passing.
    print("Testing signature..." )
    # We won't actually load the model because it takes time/memory, just rely on error if signature wrong.
    # Actually, the constructor LOADS the model. So we will fail if we run this fully.
    # But inspecting __doc__ or help is enough.
    pass
except TypeError as e:
    print(f"Signature mismatch: {e}")
