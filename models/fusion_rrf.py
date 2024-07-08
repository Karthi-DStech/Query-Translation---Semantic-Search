import json
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads


class FusionRetrieval:
    """
    This class retrieves documents using RAG Fusion queries and
    generates an answer based on the retrieved documents.

    Parameters
    ----------
    retriever:
        The retriever object that will be used to retrieve documents.

    temperature: float
        The temperature parameter for the language model.

    k: int
        The k parameter for the reciprocal rank fusion formula.

    Methods
    -------
    retrieve_documents:
        Retrieve documents using RAG Fusion queries.

    answer_question:
        Generate an answer to the question based on retrieved documents.

    run:
        Run the RAG Fusion retrieval and answering process.

    reciprocal_rank_fusion:
        Reciprocal rank fusion that takes multiple lists of ranked documents and an optional parameter k used in the RRF formula.

    """

    def __init__(self, retriever, temperature=0, k=60):
        if not retriever:
            raise ValueError("Retriever cannot be None")

        self.retriever = retriever
        self.temperature = temperature
        self.k = k
        self.method_name = "fusion"
        self.llm = ChatOpenAI(temperature=temperature)

        # Initialize prompt for generating RAG Fusion queries
        self.prompt_rag_fusion = ChatPromptTemplate.from_template(
            """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
        )

        self.generate_queries = (
            self.prompt_rag_fusion
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        # Initialize RAG prompt template
        self.rag_template = ChatPromptTemplate.from_template(
            """Answer the following question based on this context:

{context}

Question: {question}
            """
        )

    @staticmethod
    def reciprocal_rank_fusion(results: list[list], k=60):
        """
        Reciprocal rank fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula

        parameters
        ----------
        results: list[list]
            A list of lists of ranked documents.

        k: int
            The k parameter used in the RRF formula.

        returns
        -------
        reranked_results: list[tuple]
            A list of tuples containing the document and its fused score.

        raises
        ------
        ValueError
            If the results list is empty.

        """
        if not results:
            raise ValueError("Results list cannot be empty")

        try:
            # Initialize a dictionary to hold fused scores for each unique document
            fused_scores = {}

            # Iterate through each list of ranked documents
            for docs in results:

                # Iterate through each document in the list, with its rank (position in the list)
                for rank, doc in enumerate(docs):

                    # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                    doc_str = dumps(doc)

                    # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0

                    # Update the score of the document using the RRF formula: 1 / (rank + k)
                    fused_scores[doc_str] += 1 / (rank + k)

            # Sort the documents based on their fused scores in descending order to get the final reranked results
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(
                    fused_scores.items(), key=lambda x: x[1], reverse=True
                )
            ]

            # Return the reranked results as a list of tuples, each containing the document and its fused score
            return reranked_results
        except Exception as e:
            raise ValueError(f"Failed to perform reciprocal rank fusion: {str(e)}")

    def retrieve_documents(self, question: str):
        """
        Retrieve documents using RAG Fusion queries
        
        parameters
        ----------
        question: str
            The input question for which documents need to be retrieved.
            
        returns
        -------
        docs: list
            A list of documents retrieved based on the input question.
            
        raises
        ------
        ValueError
            If the question is empty or no documents are retrieved.
        """
        if not question:
            raise ValueError("Question cannot be empty")

        try:
            retrieval_chain = (
                self.generate_queries
                | self.retriever.map()
                | (lambda results: self.reciprocal_rank_fusion(results, self.k))
            )
            docs = retrieval_chain.invoke({"question": question})
            if not docs:
                raise ValueError("No documents retrieved for the given question.")
            return docs
        except Exception as e:
            raise ValueError(f"Failed to retrieve documents: {str(e)}")

    def answer_question(self, question: str, docs: list):
        """
        Generate an answer to the question based on retrieved documents
        
        parameters
        ----------
        question: str
            The input question for which an answer needs to be generated.
            
        docs: list
            A list of documents retrieved based on the input question.
            
        returns
        -------
        answer: str
            The answer generated based on the retrieved documents.
            
        raises
        ------
        ValueError
            If no documents are retrieved to generate an answer.
        """
        if not docs:
            raise ValueError("No documents retrieved to generate an answer")

        try:
            context = "\n".join([doc.page_content for doc, score in docs])
            final_rag_chain = (
                {"context": context, "question": question}
                | self.rag_template
                | self.llm
                | StrOutputParser()
            )
            return final_rag_chain.invoke({"question": question})
        except Exception as e:
            raise ValueError(f"Failed to generate answer: {str(e)}")

    def run(self):
        """
        Run the RAG Fusion retrieval and answering process
        
        parameters
        ----------
        None
        
        implements
        ----------
        retrieve_documents:
            Retrieve documents using RAG Fusion queries.
            
        answer_question:
            Generate an answer to the question based on retrieved documents.
        """
        question = input("Please enter your question: ")
        try:
            docs = self.retrieve_documents(question)
            answer = self.answer_question(question, docs)
            print("Answer:", answer)
        except ValueError as e:
            print(f"Error: {e}")
