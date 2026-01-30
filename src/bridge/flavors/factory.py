from .base import BaseFlavor
from .definitions import QwenFlavor, MiniMaxFlavor, GLMFlavor

def get_flavor_for_model(model_path: str) -> BaseFlavor:
    path_lower = model_path.lower()
    if "qwen" in path_lower: return QwenFlavor()
    if "mimo" in path_lower: return QwenFlavor() # Mimo is Qwen-compatible
    
    if "minimax" in path_lower: return MiniMaxFlavor()
    if "gpt-oss" in path_lower: return MiniMaxFlavor()
    
    if "glm" in path_lower: return GLMFlavor()
    
    return BaseFlavor()
