from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = ""
    DATABASE_URL: str = ""
    FRONTEND_URL: str = ""
    LOG_LEVEL: str = "INFO"
    GENERATOR_MODEL: str = "z-ai/glm-5.2:free"
    JUDGE_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    ENABLE_ONLINE_EVAL: bool = True
    RAG_VECTOR_THRESHOLD: float = 0.30
    EMBED_MODEL: str = "nvidia/nemotron-3-embed-1b:free"
    EMBED_DIM: int = 1024

    model_config = {"env_file": ".env"}


settings = Settings()
