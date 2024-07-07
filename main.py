from call_methods import RAGMethodCaller
from configuration import Config
from document_processor import DocumentProcessor


def main():
    """
    Execute the main function to call the RAG method based on the user's choice.

    parameters
    ----------
    None

    Process
    -------
    1. Set up configuration with API keys.
    2. Initialize DocumentProcessor and process the URL.
    3. Initialize the RAGMethodCaller with the retriever.
    4. Set the method to use (can be 'multi_query', 'fusion', 'decomposition').
    5. Prompt the user to enter the query.
    6. Call the chosen method with the provided query.

    returns
    -------
    An answer to the user's query based on the chosen RAG method.
    """
    # Set up configuration with API keys
    langchain_api_key = (
        "your_langchain_api_key_here"  # Replace with your actual Langchain API key
    )
    openai_api_key = (
        "your_openai_api_key_here"  # Replace with your actual OpenAI API key
    )
    config = Config(langchain_api_key, openai_api_key)

    if not config:
        raise ValueError("Please set the API_KEY environment variable.")

    # Initialize DocumentProcessor and process the URL
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    document_processor = DocumentProcessor(url)
    document_processor.process()
    retriever = document_processor.get_retriever()

    # Initialize the RAGMethodCaller with the retriever
    rag_method_caller = RAGMethodCaller(retriever)

    # Set the method to use (can be 'multi_query', 'fusion', 'decomposition')
    method_name = "multi_query"  # Replace with your chosen method

    # Prompt the user to enter the query
    query = input("Please enter your query: ")

    # Call the chosen method with the provided query
    rag_method_caller.call_method(method_name, query)


if __name__ == "__main__":
    main()
