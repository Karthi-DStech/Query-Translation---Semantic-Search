import os


class Config:
    """
    The Config class sets the API keys for the RAG process.

    Attributes
    ----------
    langchain_api_key: str
        The Langchain API key.
    openai_api_key: str
        The OpenAI API key.

    Methods
    -------
    setup_environment:
        Set up the environment for the RAG process.
    """

    def __init__(self, langchain_api_key, openai_api_key):
        self.langchain_api_key = langchain_api_key
        self.openai_api_key = openai_api_key
        self.setup_environment()

    def setup_environment(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
