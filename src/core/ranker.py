from sentence_transformers import CrossEncoder

class RankerService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Singleton: Load model once
            print("Loading Cross-Encoder Model (this takes a moment)...")
            cls._instance = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return cls._instance

    @staticmethod
    def rank(query, docs, top_k=5):
        """
        Ranks a list of documents based on relevance to the query.
        Args:
            query (str): The user query.
            docs (list): List of document objects (must have .content attribute).
            top_k (int): Number of top results to return.
        Returns:
            list: The top_k ranked documents.
        """
        model = RankerService.get_instance()
        if not docs: return []
        
        # Prepare pairs for CrossEncoder [query, doc_text]
        # Assuming 'doc' has a 'content' attribute which is the text
        pairs = [[query, doc.content] for doc in docs]
        
        # CrossEncoder returns a list of float scores for each pair
        scores = model.predict(pairs)
        
        # Combine docs with their scores
        doc_scores = zip(docs, scores)
        
        # Sort by score descending
        ranked = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Return only the doc objects
        return [doc for doc, score in ranked][:top_k]
