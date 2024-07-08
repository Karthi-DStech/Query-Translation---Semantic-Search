from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class DecompositionRetrieval:
    """
    This class retrieves documents using RAG Decomposition queries and 
    generates an answer based on the retrieved documents.

    Parameters
    ----------
    retriever:
        The retriever object that will be used to retrieve documents.

    temperature: float
        The temperature parameter for the language model.

    Methods
    -------

    retrieve_and_rag:
        RAG on each sub-question.

    format_qa_pairs:
        Format Q and A pairs.

    run:
        Run the decomposition retrieval and answering process.

    """

    def __init__(self, retriever, temperature=0):
        if not retriever:
            raise ValueError("Retriever cannot be None")

        self.retriever = retriever
        self.temperature = temperature
        self.method_name = "decomposition"
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)

        # Initialize prompt for generating sub-questions
        self.prompt_decomposition = ChatPromptTemplate.from_template(
            """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
        )

        self.generate_queries_decomposition = (
            self.prompt_decomposition
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        # Initialize decomposition prompt template
        self.decomposition_prompt = ChatPromptTemplate.from_template(
            """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
            """
        )

    @staticmethod
    def format_qa_pair(question, answer):
        """
        Format Q and A pair using decomposition prompt template.

        Parameters
        ----------
        question: str
            The question.

        answer: str
            The answer.

        Returns
        -------
        str
            The formatted Q and A pair.
        """

        return f"Question: {question}\nAnswer: {answer}\n\n".strip()

    def retrieve_and_rag(self, question, sub_question_generator_chain):
        """
        RAG on each sub-question using decomposition prompt template and answer the main question sequentially.

        Parameters
        ----------
        question: str
            The question.

        sub_question_generator_chain: Chain
            The chain to generate sub-questions.

        Returns
        -------
        list
            A list of RAG chain results.

        list
            A list of sub-questions.

        Raises
        ------
        ValueError
            If question is empty.
        """
        if not question:
            raise ValueError("Question cannot be empty")

        try:
            # Use decomposition chain to generate sub-questions
            sub_questions = sub_question_generator_chain.invoke({"question": question})

            # Initialize a list to hold RAG chain results
            rag_results = []

            for sub_question in sub_questions:
                
                # Retrieve documents for each sub-question
                retrieved_docs = self.retriever.get_relevant_documents(sub_question)

                # Use retrieved documents and sub-question in RAG chain
                answer = (
                    self.decomposition_prompt | self.llm | StrOutputParser()
                ).invoke({"context": retrieved_docs, "question": sub_question})
                rag_results.append(answer)

            return rag_results, sub_questions
        except Exception as e:
            raise ValueError(f"Failed to retrieve and RAG sub-questions: {str(e)}")

    def format_qa_pairs(self, questions, answers):
        """
        Format Q and A pairs in the template for passing to the LLM for final answer generation.
        
        Parameters
        ----------
        questions: list
            A list of questions.
            
        answers: list
            A list of answers.
            
        Returns
        -------
        str
            The formatted Q and A pairs.
            
        Raises
        ------
        ValueError
            If failed to format Q and A pairs.
        """
        try:
            formatted_string = ""
            for i, (question, answer) in enumerate(zip(questions, answers), start=1):
                formatted_string += (
                    f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
                )
            return formatted_string.strip()
        except Exception as e:
            raise ValueError(f"Failed to format Q and A pairs: {str(e)}")

    def run(self):
        """
        Run the decomposition retrieval and answering process
        
        parameters
        ----------
        None
        
        implements
        ----------
        retrieve_and_rag:
            RAG on each sub-question.
            
        format_qa_pairs:
            Format Q and A pairs.
            
        """
        question = input("Please enter your question: ")
        try:
            answers, sub_questions = self.retrieve_and_rag(
                question, self.generate_queries_decomposition
            )
            context = self.format_qa_pairs(sub_questions, answers)

            final_prompt = ChatPromptTemplate.from_template(
                """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
                """
            )

            final_rag_chain = final_prompt | self.llm | StrOutputParser()
            final_answer = final_rag_chain.invoke(
                {"context": context, "question": question}
            )
            print("Final Answer:", final_answer)
        except ValueError as e:
            print(f"Error: {e}")
