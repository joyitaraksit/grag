import os
import pickle
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import logging

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, db_dir: str = "graphrag_db"):
        self.db_dir = db_dir
        self.entity_graph = nx.Graph()
        self.vector_store = None
        self.embeddings = None
        os.makedirs(db_dir, exist_ok=True)
        
    def initialize_embeddings(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            test_embedding = self.embeddings.embed_query("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding model failed to generate test embedding")
            logger.info(f"Successfully initialized embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
        
    def _get_email_address(self, email_field: Union[str, List[str]]) -> str:
        if isinstance(email_field, list):
            return email_field[0] if email_field else "unknown"
        return email_field if email_field else "unknown"
        
    def build_integrated_graph(self, emails: List[Dict]) -> nx.Graph:
        self.entity_graph = nx.Graph()
        person_topics = {}
        topic_docs = {}
        
        for email in emails:
            doc_id = email['id']
            sender = self._get_email_address(email['from'])
            recipient = self._get_email_address(email['to'])
            
            doc = Document(
                page_content=email['message'],
                metadata={
                    'id': doc_id,
                    'from': sender,
                    'to': recipient,
                    'subject': email['subject'],
                    'date': email['date']
                }
            )
            
            topics = self._extract_topics(doc)
            
            for person in [sender, recipient]:
                if person not in person_topics:
                    person_topics[person] = set()
                person_topics[person].update(topics)
            
            for topic in topics:
                if topic not in topic_docs:
                    topic_docs[topic] = []
                topic_docs[topic].append((doc_id, 1.0))
            
            self.entity_graph.add_node(
                doc_id,
                type='document',
                content=email['message'],
                metadata=doc.metadata
            )
            
            for topic in topics:
                self.entity_graph.add_edge(
                    doc_id, topic,
                    type='contains',
                    weight=1.0
                )
        
        for person, topics in person_topics.items():
            self.entity_graph.add_node(
                person,
                type='person',
                topics=list(topics)
            )
            
            for topic in topics:
                if self.entity_graph.has_edge(person, topic):
                    self.entity_graph[person][topic]['weight'] += 1.0
                else:
                    self.entity_graph.add_edge(
                        person, topic,
                        type='discusses',
                        weight=1.0
                    )
        
        graph_path = os.path.join(self.db_dir, "entity_graph.pkl")
        with open(graph_path, 'wb') as f:
            pickle.dump(self.entity_graph, f)
        logger.info(f"Saved integrated graph with {len(self.entity_graph.nodes)} nodes and {len(self.entity_graph.edges)} edges")
        
        return self.entity_graph
    
    def _extract_topics(self, doc: Document) -> List[str]:
        text = doc.page_content.lower()
        topics = set()
        
        for delimiter in [',', '.', ';', ':', '!', '?', '\n']:
            parts = text.split(delimiter)
            for part in parts:
                topic = part.strip()
                if topic and len(topic) > 3:
                    topics.add(topic)
        
        return list(topics)
    
    def _convert_metadata_to_strings(self, metadata: Dict) -> Dict:
        converted = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                converted[key] = ", ".join(str(v) for v in value)
            else:
                converted[key] = str(value)
        return converted
        
    def build_vector_store(self, emails: List[Dict]):
        if not self.embeddings:
            self.initialize_embeddings()
            
        documents = []
        for email in emails:
            if not email.get('message', '').strip():
                logger.warning(f"Skipping email {email.get('id')} with empty message")
                continue
                
            metadata = {
                'id': email['id'],
                'from': email['from'],
                'to': email['to'],
                'subject': email['subject'],
                'date': email['date']
            }
            metadata = self._convert_metadata_to_strings(metadata)
            
            doc = Document(
                page_content=email['message'],
                metadata=metadata
            )
            documents.append(doc)
        
        if not documents:
            raise ValueError("No valid documents to add to vector store")
            
        try:
            vector_store_path = os.path.join(self.db_dir, "chroma_db")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=vector_store_path
            )
            
            self.vector_store.persist()
            logger.info(f"Successfully saved vector store with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
    
    def load_graph(self) -> Optional[nx.Graph]:
        graph_path = os.path.join(self.db_dir, "entity_graph.pkl")
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.entity_graph = pickle.load(f)
            logger.info(f"Loaded graph with {len(self.entity_graph.nodes)} nodes and {len(self.entity_graph.edges)} edges")
            return self.entity_graph
        return None
    
    def load_vector_store(self):
        vector_store_path = os.path.join(self.db_dir, "chroma_db")
        if os.path.exists(vector_store_path):
            if not self.embeddings:
                self.initialize_embeddings()
            self.vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=self.embeddings
            )
            logger.info("Loaded vector store")