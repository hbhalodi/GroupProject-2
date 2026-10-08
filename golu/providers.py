"""Provider abstraction: one function that returns a chat model for any provider.

Owner: Member B

The rest of Golu only talks to the LangChain chat-model interface
(`bind_tools`, `astream`), so switching between a local Ollama model and a
cloud model is just a different --provider / --model flag.

Provider packages are imported lazily, so a teammate who only uses Ollama
doesn't need the Groq/OpenAI/Anthropic packages installed.
"""

from __future__ import annotations

import os

from golu.config import Settings

SUPPORTED_PROVIDERS = ("ollama", "groq", "openai", "anthropic")

# Environment variable each cloud provider needs.
_API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ProviderError(RuntimeError):
    """Raised when a provider can't be created (missing package or API key)."""


def create_chat_model(settings: Settings):
    """Build a LangChain chat model for settings.provider / settings.model.

    Returns:
        A LangChain BaseChatModel that supports tool calling.
    """
    provider = settings.provider.lower()
    model = settings.resolved_model()

    if provider not in SUPPORTED_PROVIDERS:
        raise ProviderError(
            f"Unknown provider '{provider}'. Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    key_env = _API_KEY_ENV.get(provider)
    if key_env and not os.getenv(key_env):
        raise ProviderError(f"{key_env} is not set. Add it to your .env file.")

    try:
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model,
                temperature=settings.temperature,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                # A larger context window helps with file contents + tool results.
                num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
            )

        if provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(model=model, temperature=settings.temperature)

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, temperature=settings.temperature)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, temperature=settings.temperature)

    except ImportError as exc:
        package = {
            "ollama": "langchain-ollama",
            "groq": "langchain-groq",
            "openai": "langchain-openai",
            "anthropic": "langchain-anthropic",
        }[provider]
        raise ProviderError(f"Provider '{provider}' needs: pip install {package}") from exc

    raise ProviderError(f"Provider '{provider}' is not implemented")  # pragma: no cover
