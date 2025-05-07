import networkx as nx
from typing import List, Tuple, Dict, Any
from langchain.docstore.document import Document
import logging

logger = logging.getLogger(__name__)

class GraphRetriever:
    def __init__(self, vector_store, entity_graph: nx.Graph):
        self.vector_store = vector_store
        self.entity_graph = entity_graph
        
    def vector_retrieval(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def topic_enhanced_retrieval(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        initial_results = self.vector_store.similarity_search_with_score(query, k=k*3)
        
        query_terms = query.lower().split()
        
        scored_docs = []
        for doc, vector_score in initial_results:
            doc_id = doc.metadata['id']
            topic_score = 0.0
            
            if doc_id in self.entity_graph:
                doc_topics = []
                for neighbor in self.entity_graph.neighbors(doc_id):
                    if self.entity_graph.nodes[neighbor].get('type') == 'topic':
                        doc_topics.append(neighbor)
                
                for topic in doc_topics:
                    for term in query_terms:
                        if term in topic.lower():
                            topic_score += 1.0
                            
                            if term == topic.lower():
                                topic_score += 0.5
                
                if doc_topics:
                    topic_score /= len(doc_topics)
            
            final_score = (0.7 * vector_score) + (0.3 * topic_score)
            scored_docs.append((doc, final_score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]
    
    def get_related_topics(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_terms = query.lower().split()
        topic_scores = {}
        
        for node in self.entity_graph.nodes():
            if self.entity_graph.nodes[node].get('type') == 'topic':
                score = 0.0
                for term in query_terms:
                    if term in node.lower():
                        score += 1.0
                        if term == node.lower():
                            score += 0.5
                if score > 0:
                    topic_scores[node] = score
        
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:k]
    
    def get_related_people(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_terms = query.lower().split()
        person_scores = {}
        
        for node in self.entity_graph.nodes():
            if self.entity_graph.nodes[node].get('type') == 'person':
                score = 0.0
                topics = self.entity_graph.nodes[node].get('topics', [])
                
                for topic in topics:
                    for term in query_terms:
                        if term in topic.lower():
                            score += 1.0
                            if term == topic.lower():
                                score += 0.5
                
                if score > 0:
                    person_scores[node] = score
        
        sorted_people = sorted(person_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_people[:k]