from models.multi_query import MultiQueryRetrieval
from models.fusion_rrf import FusionRetrieval
from models.decomposition_ltm import DecompositionRetrieval

class RAGMethodCaller:
    """
    This class is responsible for calling the different RAG methods based on the user's choice.
    
    Args
    ----
    retriever:
        The retriever object that will be used to retrieve documents.
        
    Methods
    -------
    call_method(method_name):
        Calls the specified RAG method based on the method name provided.
    """
    def __init__(self, retriever):
        """
        Initializes the RAGMethodCaller with the retriever object.
        
        parameters
        ----------
        retriever:
            The retriever object that will be used to retrieve documents.
            
        Raises
        ------
        ValueError:
            If the retriever object is None.
        """
        if not retriever:
            raise ValueError("Retriever cannot be None")
        self.retriever = retriever

    def call_method(self, method_name):
        """
        Calls the specified RAG method based on the method name provided.
        
        parameters
        ----------
        method_name: str
            The name of the method to be called.
            
        raises
        ------
        ValueError:
            If the method name is invalid.
        """
        if method_name.lower() == "multi_query":
            try:
                multi_query_retrieval = MultiQueryRetrieval(self.retriever)
                multi_query_retrieval.run()
            except ValueError as e:
                print(f"Error in MultiQueryRetrieval: {e}")

        elif method_name.lower() == "fusion":
            try:
                fusion_retrieval = FusionRetrieval(self.retriever)
                fusion_retrieval.run()
            except ValueError as e:
                print(f"Error in FusionRetrieval: {e}")

        elif method_name.lower() == "decomposition":
            try:
                decomposition_retrieval = DecompositionRetrieval(self.retriever)
                decomposition_retrieval.run()
            except ValueError as e:
                print(f"Error in DecompositionRetrieval: {e}")

        else:
            print("Invalid method name provided. Please choose from 'multi_query', 'fusion', or 'decomposition'.")

