# Import necessary libraries
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class DocumentProcessor:
    """
    The DocumentProcessor class processes a given URL by loading the documents, splitting them into chunks,
    indexing the chunks, and providing a retriever for the indexed documents.
    
    Parameters
    ----------
    url: str
        The URL to process.
    
    chunk_size: int
        The size of each chunk of text.
        
    chunk_overlap: int
        The overlap between chunks.
        
    parse_classes: tuple
        The classes to parse from the HTML.
        
    Methods
    -------
    load_documents:
        Load the documents from the URL.
        
    split_documents:
        Split the documents into chunks.
        
    index_documents:
        Index the documents.
        
    process:
        Process the documents.
        
    get_retriever:
        Get the retriever for the indexed documents.
    """
    def __init__(
        self,
        url,
        chunk_size=300,
        chunk_overlap=50,
        parse_classes=("post-content", "post-title", "post-header"),
    ):
        self.url = url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parse_classes = parse_classes
        self.documents = None
        self.splits = None
        self.vectorstore = None
        self.retriever = None

    def load_documents(self):
        """
        load_documents method loads the documents from the URL.
        
        parameters
        ----------
        None
        
        implements
        ----------
        loader: WebBaseLoader
            The WebBaseLoader object to load the documents.
            
        documents: list
            The list of documents loaded from the URL.

        """
        loader = WebBaseLoader(
            web_paths=(self.url,),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=self.parse_classes)),
        )
        self.documents = loader.load()

    def split_documents(self):
        """
        Split the documents into chunks.
        
        parameters
        ----------
        None
        
        implements
        ----------
        text_splitter: RecursiveCharacterTextSplitter
            The RecursiveCharacterTextSplitter object to split the documents.
            
        splits: list
            The list of splits of the documents.
            
        raises
        ------
        ValueError
            If the documents are not loaded.
        """
        if not self.documents:
            raise ValueError("Documents not loaded. Call load_documents() first.")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.splits = text_splitter.split_documents(self.documents)

    def index_documents(self):
        """
        Index the documents.
        
        parameters
        ----------
        None
        
        implements
        ----------
        vectorstore: Chroma
            The Chroma object to index the documents.
            
        retriever: Retriever
            The retriever for the indexed documents.
            
        raises
        ------
        ValueError
            If the documents are not split.
        """
        if not self.splits:
            raise ValueError("Documents not split. Call split_documents() first.")
        self.vectorstore = Chroma.from_documents(
            documents=self.splits, embedding=OpenAIEmbeddings()
        )
        self.retriever = self.vectorstore.as_retriever()

    def process(self):
        """
        Process the documents.
        
        parameters
        ----------
        None
        
        implements
        ----------
        load_documents:
            Load the documents.
        
        split_documents:
            Split the documents.
            
        index_documents:
            Index the documents.
        """
        self.load_documents()
        self.split_documents()
        self.index_documents()

    def get_retriever(self):
        """
        returns the retriever for the indexed documents.
        
        parameters
        ----------
        None
        
        returns
        -------
        retriever: Retriever
            The retriever for the indexed documents.
            
        raises
        ------
        ValueError
            If the documents are not indexed.
        """
        if not self.retriever:
            raise ValueError("Documents not indexed. Call process() first.")
        return self.retriever

