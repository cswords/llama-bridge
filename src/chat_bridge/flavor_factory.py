from typing import Dict, Any, Type
import logging
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase
from src.chat_bridge.flavors.qwen_flavor import QwenFlavor
from src.chat_bridge.flavors.glm_flavor import GLMFlavor
from src.chat_bridge.flavors.gpt_oss_flavor import GPTOSSFlavor
from src.chat_bridge.flavors.minimax_flavor import MiniMaxFlavor
from src.chat_bridge.flavors.mimo_flavor import MimoFlavor

logger = logging.getLogger(__name__)

class FlavorFactory:
    """
    Factory to auto-detect and instantiate the correct ChatFlavorBase
    based on model path or name.
    """
    
    @staticmethod
    def get_flavor_for_model(model_path: str) -> ChatFlavorBase:
        """
        Heuristic detection of flavor from model path string.
        """
        model_path_lower = model_path.lower()
        
        if "qwen" in model_path_lower:
            logger.info(f"FlavorFactory: Detected QwenFlavor for model {model_path}")
            return QwenFlavor()
        elif "glm" in model_path_lower:
            logger.info(f"FlavorFactory: Detected GLMFlavor for model {model_path}")
            return GLMFlavor()
        elif "minimax" in model_path_lower:
            logger.info(f"FlavorFactory: Detected MiniMaxFlavor for model {model_path}")
            return MiniMaxFlavor()
        elif "mimo" in model_path_lower:
            logger.info(f"FlavorFactory: Detected MimoFlavor for model {model_path}")
            return MimoFlavor()
        elif "gpt-oss" in model_path_lower or "harmony" in model_path_lower:
            logger.info(f"FlavorFactory: Detected GPTOSSFlavor for model {model_path}")
            return GPTOSSFlavor()
        
        # Default Fallback
        logger.warning(f"FlavorFactory: No specific flavor matched for {model_path}. Defaulting to QwenFlavor (Generic XML).")
        return QwenFlavor()

