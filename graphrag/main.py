import os
import argparse
import logging
from typing import List, Tuple
from langchain.docstore.document import Document

from graphrag.core.graph_builder import GraphBuilder
from graphrag.retrieval.retriever import GraphRetriever
from graphrag.models.email import EmailCollection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphRAG:
    def __init__(self, db_dir: str = "graphrag_db"):
        self.db_dir = db_dir
        self.graph_builder = GraphBuilder(db_dir)
        self.email_collection = EmailCollection(os.path.join(db_dir, "emails.json"))
        self.retriever = None
    
    def populate_database(self, use_mock_data: bool = True):
        self.email_collection.load()
        
        self.graph_builder.build_integrated_graph(self.email_collection.get_emails())
        self.graph_builder.build_vector_store(self.email_collection.get_emails())
        
        self.retriever = GraphRetriever(
            self.graph_builder.vector_store,
            self.graph_builder.entity_graph
        )
    
    def run_demo(self, query: str, k: int = 5):
        if not self.retriever:
            logger.error("Database not populated. Run populate_database() first.")
            return
        
        vector_results = self.retriever.vector_retrieval(query, k)
        enhanced_results = self.retriever.topic_enhanced_retrieval(query, k)
        
        related_topics = self.retriever.get_related_topics(query, k)
        related_people = self.retriever.get_related_people(query, k)
        
        print("\n=== Standard Vector Retrieval ===")
        self._display_results(vector_results)
        
        print("\n=== Topic-Enhanced Retrieval ===")
        self._display_results(enhanced_results)
        
        print("\n=== Related Topics ===")
        for topic, score in related_topics:
            print(f"- {topic} (relevance: {score:.2f})")
        
        print("\n=== Related People ===")
        for person, score in related_people:
            print(f"- {person} (relevance: {score:.2f})")
    
    def _display_results(self, results: List[Tuple[Document, float]]):
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.2f}")
            print(f"From: {doc.metadata['from']}")
            print(f"To: {doc.metadata['to']}")
            print(f"Subject: {doc.metadata['subject']}")
            print(f"Date: {doc.metadata['date']}")
            print(f"Message: {doc.page_content[:200]}...")

def main():
    parser = argparse.ArgumentParser(description="GraphRAG - Graph-based RAG System")
    parser.add_argument("--db-dir", default="graphrag_db", help="Database directory")
    parser.add_argument("--query", help="Search query for demo")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--use-mock", action="store_true", help="Use mock data")
    
    args = parser.parse_args()
    
    graphrag = GraphRAG(args.db_dir)
    graphrag.populate_database(use_mock_data=args.use_mock)
    
    if args.query:
        graphrag.run_demo(args.query, args.k)
    else:
        graphrag.run_demo("project database schema", args.k)

if __name__ == "__main__":
    main() 