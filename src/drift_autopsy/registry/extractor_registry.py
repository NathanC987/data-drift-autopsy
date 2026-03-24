"""Central registry for embedding extractors."""

from typing import Any, Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """Registry for extractor discovery and instantiation."""

    _extractors: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an extractor class."""

        def decorator(extractor_class: Type) -> Type:
            if name in cls._extractors:
                logger.warning("Extractor '%s' is already registered. Overwriting.", name)

            cls._extractors[name] = extractor_class
            logger.debug("Registered extractor: %s -> %s", name, extractor_class.__name__)
            return extractor_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any):
        """Factory method to create an extractor instance."""
        if name not in cls._extractors:
            available = ", ".join(cls.list())
            raise ValueError(f"Unknown extractor: '{name}'. Available extractors: {available}")

        extractor_class = cls._extractors[name]
        return extractor_class(**kwargs)

    @classmethod
    def list(cls) -> list:
        """List all registered extractor names."""
        return list(cls._extractors.keys())

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get extractor class by name."""
        return cls._extractors.get(name)

    @classmethod
    def clear(cls):
        """Clear all registered extractors."""
        cls._extractors.clear()
