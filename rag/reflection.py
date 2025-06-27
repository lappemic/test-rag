"""
Reflection pattern implementation for the Swiss Legal Chatbot RAG system.
(idea from https://www.philschmid.de/agentic-pattern)
"""
import logging
import re
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class EvaluationStatus(Enum):
    """Status of the reflection evaluation."""
    PASS = "PASS"
    FAIL = "FAIL"


class ReflectionEvaluation(BaseModel):
    """Evaluation result from the reflector."""
    evaluation: EvaluationStatus
    feedback: str
    reasoning: str
    mentioned_sources: List[str]


class RAGReflector:
    """Implements reflection pattern for RAG responses."""
    
    def __init__(self, llm, embeddings, rag_retriever, max_iterations: int = 3):
        """
        Initialize the reflector.
        
        Args:
            llm: Language model for evaluation
            embeddings: Embeddings model
            rag_retriever: RAGRetriever instance for additional searches
            max_iterations: Maximum number of reflection iterations
        """
        self.llm = llm
        self.embeddings = embeddings
        self.rag_retriever = rag_retriever
        self.max_iterations = max_iterations
        self._setup_evaluation_prompt()
        self._setup_refinement_prompt()
    
    def _setup_evaluation_prompt(self):
        """Set up the evaluation prompt template with structured output."""
        # Set up the parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=ReflectionEvaluation)
        
        self.evaluation_prompt = ChatPromptTemplate.from_template(
            """You are a Swiss legal expert evaluator. Your task is to analyze a RAG response and determine if it mentions other legal sources, articles, or documents that should be searched for to provide a more complete answer.

            Analyze the following RAG response and determine:
            1. Does it mention specific legal documents, articles, or provisions that are NOT already included in the retrieved sources?
            2. Are there references to other laws, regulations, or legal principles that could provide additional relevant information?
            3. Does the response suggest that additional information might be needed from other sources?

            ORIGINAL QUERY: {original_query}

            RAG RESPONSE: {response}

            RETRIEVED SOURCES: {sources_summary}

            Evaluation Criteria:
            - PASS: The response is complete and doesn't mention additional sources that need to be searched
            - FAIL: The response mentions specific legal references, documents, or provisions that should be searched for additional information

            If FAIL, list the specific sources/documents/articles mentioned that should be searched for.

            {format_instructions}
            """
                    )
    
    def _setup_refinement_prompt(self):
        """Set up the refinement prompt template."""
        self.refinement_prompt = ChatPromptTemplate.from_template(
            """You are a Swiss legal expert. You previously provided a response to a legal question, and now you have additional relevant information from related sources. Your task is to refine and improve your original response by incorporating the new information.

            ORIGINAL QUERY: {original_query}

            ORIGINAL RESPONSE: {original_response}

            ADDITIONAL INFORMATION FROM NEW SOURCES: {additional_context}

            FEEDBACK FOR IMPROVEMENT: {feedback}

            Please provide a refined response that:
            1. Incorporates the additional relevant information
            2. Maintains all the original accuracy and citations
            3. Provides a more comprehensive and complete answer
            4. Follows the same format and style as the original response
            5. Includes proper citations for both original and new sources

            Ensure the refined response is coherent and well-structured, seamlessly integrating the new information with the original content.
            """
                    )
    
    def evaluate_response(self, original_query: str, response: str, sources: List[str]) -> ReflectionEvaluation:
        """
        Evaluate a RAG response to determine if additional sources should be searched.
        
        Args:
            original_query: The original user query
            response: The RAG response to evaluate
            sources: List of sources used in the original response
            
        Returns:
            ReflectionEvaluation with assessment and mentioned sources
        """
        sources_summary = "\n".join([f"- {src[:200]}..." for src in sources[:5]])
        
        try:
            # Try structured output first
            evaluation_chain = (
                self.evaluation_prompt 
                | self.llm 
                | self.parser
            )
            
            result = evaluation_chain.invoke({
                "original_query": original_query,
                "response": response,
                "sources_summary": sources_summary,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return result
            
        except Exception as e:
            logging.warning(f"Structured parsing failed, falling back to manual parsing: {e}")
            
            # Fallback to string parsing
            try:
                evaluation_chain = self.evaluation_prompt | self.llm | StrOutputParser()
                result_str = evaluation_chain.invoke({
                    "original_query": original_query,
                    "response": response,
                    "sources_summary": sources_summary,
                    "format_instructions": self.parser.get_format_instructions()
                })
                
                return self._parse_evaluation_manually(result_str)
                
            except Exception as e2:
                logging.error(f"Error in response evaluation: {e2}")
                # Return PASS on error to avoid infinite loops
                return ReflectionEvaluation(
                    evaluation=EvaluationStatus.PASS,
                    feedback="Evaluation failed, accepting current response",
                    reasoning=f"Error during evaluation: {str(e2)}",
                    mentioned_sources=[]
                )
    
    def _parse_evaluation_manually(self, result_str: str) -> ReflectionEvaluation:
        """Manually parse evaluation result as fallback."""
        lines = result_str.strip().split('\n')
        evaluation_status = EvaluationStatus.PASS
        feedback = ""
        reasoning = ""
        mentioned_sources = []
        
        # Simple parsing logic
        for line in lines:
            line = line.strip()
            if '"evaluation":' in line or 'evaluation:' in line:
                if 'FAIL' in line.upper():
                    evaluation_status = EvaluationStatus.FAIL
            elif '"mentioned_sources":' in line or 'mentioned_sources:' in line:
                # Try to extract sources from the line
                if '[' in line and ']' in line:
                    sources_part = line[line.find('['):line.find(']')+1]
                    try:
                        # Simple extraction of quoted strings
                        import re
                        sources = re.findall(r'"([^"]*)"', sources_part)
                        mentioned_sources.extend(sources)
                    except:
                        pass
        
        # If we found FAIL but no sources, try to extract from the entire text
        if evaluation_status == EvaluationStatus.FAIL and not mentioned_sources:
            # Look for patterns that might indicate legal references
            legal_patterns = [
                r'Art\.?\s*\d+',  # Article references
                r'SR\s*\d+',      # SR numbers
                r'Artikel\s*\d+', # German Article
                r'§\s*\d+',       # Section symbols
            ]
            
            for pattern in legal_patterns:
                matches = re.findall(pattern, result_str, re.IGNORECASE)
                mentioned_sources.extend(matches)
        
        return ReflectionEvaluation(
            evaluation=evaluation_status,
            feedback=feedback or "Manual parsing result",
            reasoning=reasoning or "Parsed manually due to structured parsing failure",
            mentioned_sources=list(set(mentioned_sources))  # Remove duplicates
        )
    
    def search_additional_sources(self, mentioned_sources: List[str], collections, conversation_history=None) -> Tuple[List[str], List[Dict]]:
        """
        Search for additional sources mentioned in the evaluation.
        
        Args:
            mentioned_sources: List of sources to search for
            collections: ChromaDB collections to search
            conversation_history: Conversation history for context
            
        Returns:
            Tuple of (retrieved_docs, retrieved_metas)
        """
        all_docs = []
        all_metas = []
        
        for source in mentioned_sources:
            try:
                # Use the RAG retriever to search for the mentioned source
                response, docs, metas = self.rag_retriever.query_rag(
                    source, collections, num_results=3, conversation_history=conversation_history
                )
                all_docs.extend(docs)
                all_metas.extend(metas)
                logging.info(f"Found {len(docs)} additional documents for source: {source}")
            except Exception as e:
                logging.error(f"Error searching for additional source '{source}': {e}")
        
        return all_docs, all_metas
    
    def refine_response(self, original_query: str, original_response: str, additional_docs: List[str], additional_metas: List[Dict], feedback: str) -> str:
        """
        Refine the original response with additional information.
        
        Args:
            original_query: The original user query
            original_response: The original RAG response
            additional_docs: Additional retrieved documents
            additional_metas: Additional retrieved metadata
            feedback: Feedback from the evaluation
            
        Returns:
            Refined response incorporating additional information
        """
        if not additional_docs:
            return original_response
        
        additional_context = "\n".join([
            f"[Source: {meta.get('document_title', 'Unknown')} - {meta.get('article_id', 'Unknown') or 'meta'}]\n{doc}"
            for doc, meta in zip(additional_docs, additional_metas)
        ])
        
        try:
            refinement_chain = self.refinement_prompt | self.llm | StrOutputParser()
            refined_response = refinement_chain.invoke({
                "original_query": original_query,
                "original_response": original_response,
                "additional_context": additional_context,
                "feedback": feedback
            })
            
            return refined_response.strip()
            
        except Exception as e:
            logging.error(f"Error refining response: {e}")
            return original_response  # Return original on error
    
    def reflect_and_refine(self, original_query: str, initial_response: str, initial_sources: List[str], initial_metas: List[Dict], collections, conversation_history=None) -> Tuple[str, Dict]:
        """
        Apply reflection pattern to iteratively improve the RAG response.
        
        Args:
            original_query: The original user query
            initial_response: Initial RAG response
            initial_sources: Initial retrieved documents
            initial_metas: Initial retrieved metadata
            collections: ChromaDB collections for additional searches
            conversation_history: Conversation history for context
            
        Returns:
            Tuple of (final_response, reflection_info)
        """
        current_response = initial_response
        all_docs = initial_sources[:]
        all_metas = initial_metas[:]
        
        reflection_info = {
            "iterations": 0,
            "evaluations": [],
            "additional_sources_found": 0,
            "final_status": "no_reflection_needed"
        }
        
        for iteration in range(self.max_iterations):
            logging.info(f"Reflection iteration {iteration + 1}/{self.max_iterations}")
            
            # Evaluate current response
            evaluation = self.evaluate_response(original_query, current_response, all_docs)
            reflection_info["evaluations"].append({
                "iteration": iteration + 1,
                "status": evaluation.evaluation.value,
                "feedback": evaluation.feedback,
                "mentioned_sources": evaluation.mentioned_sources
            })
            
            # If evaluation passes or no mentioned sources, we're done
            if evaluation.evaluation == EvaluationStatus.PASS or not evaluation.mentioned_sources:
                reflection_info["final_status"] = "completed_successfully"
                break
            
            # Search for additional sources
            additional_docs, additional_metas = self.search_additional_sources(
                evaluation.mentioned_sources, collections, conversation_history
            )
            
            if additional_docs:
                reflection_info["additional_sources_found"] += len(additional_docs)
                all_docs.extend(additional_docs)
                all_metas.extend(additional_metas)
                
                # Refine response with additional information
                current_response = self.refine_response(
                    original_query, current_response, additional_docs, additional_metas, evaluation.feedback
                )
                
                logging.info(f"Response refined with {len(additional_docs)} additional sources")
            else:
                logging.info("No additional sources found, stopping reflection")
                reflection_info["final_status"] = "no_additional_sources_found"
                break
            
            reflection_info["iterations"] = iteration + 1
        
        if reflection_info["iterations"] == self.max_iterations:
            reflection_info["final_status"] = "max_iterations_reached"
        
        logging.info(f"Reflection completed after {reflection_info['iterations']} iterations")
        return current_response, reflection_info 