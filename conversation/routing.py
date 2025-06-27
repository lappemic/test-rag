"""
Query routing for the Swiss Legal Chatbot.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch


class QueryRouter:
    """Routes user queries to appropriate handlers."""
    
    def __init__(self, llm):
        """Initialize with language model for routing decisions."""
        self.llm = llm
        self._setup_router()
    
    def _setup_router(self):
        """Set up the routing prompt and chain."""
        routing_prompt_template = ChatPromptTemplate.from_template(
            """You are an expert at routing a user question to a vectorstore or a special function.

            The vectorstore contains documents about Swiss law and can answer specific legal questions.

            The special function 'list_known_laws' can list all the laws the chatbot knows about. It should be used for questions like "What laws do you know?", "Welche Gesetze kannst du beantworten?", or "Welche Gesetze kennst du?".

            For which of these two options is the user question asking for?

            Respond with a single word: "vectorstore" or "list_known_laws".

            User Question: {question}"""
        )
        
        self.router = routing_prompt_template | self.llm | StrOutputParser()
    
    def route_query(self, question):
        """Route a user question to the appropriate handler."""
        return self.router.invoke({"question": question})
    
    def create_routing_chain(self, list_laws_func, rag_func):
        """
        Create a routing chain that branches based on query type.
        
        Args:
            list_laws_func: Function to handle law listing queries
            rag_func: Function to handle RAG queries
        
        Returns:
            A RunnableBranch chain for query routing
        """
        branch = RunnableBranch(
            (lambda x: "list_known_laws" in x["topic"].lower(), list_laws_func),
            rag_func
        )
        
        return {"topic": self.router, "question": lambda x: x["question"]} | branch 