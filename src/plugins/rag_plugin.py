# src/plugins/rag_plugin.py
from semantic_kernel.functions import kernel_function
from src.core.kernel import get_collection

class AdvancedRagPlugin:
    def __init__(self, embedding_service, config):
        self.collection = get_collection()
        self.embedding_gen = embedding_service
        self.config = config

    @kernel_function(name="SearchKnowledge")
    async def search(self, query: str) -> str:
        """
        Searches for textual information, policies, reports, and qualitative content in the PDF knowledge base.
        """
        from src.core.ranker import RankerService

        # 1. Convert User Query to Vector
        print(f"DEBUG: RAG Search Query -> '{query}'")
        query_vectors = await self.embedding_gen.generate_embeddings([query])
        query_vector = query_vectors[0]
        
        # 2. Search Collection (Get top K for re-ranking)
        results = await self.collection.search(
            vector=query_vector,
            top=self.config["rag"]["retrieve_top_k"] 
        )

        # Extract Records from Search Results
        initial_docs = []
        if results and results.results:
            async for result in results.results:
                initial_docs.append(result.record)
        
        print(f"DEBUG: ChromaDB returned {len(initial_docs)} raw results.")

        # 3. Re-Rank Results using Cross-Encoder
        # RankerService takes (query, docs, top_k)
        ranked_docs = RankerService.rank(query, initial_docs, top_k=5)
        print(f"DEBUG: Ranker kept {len(ranked_docs)} results.")
        
        if not ranked_docs:
            print("DEBUG: Ranker filtered out ALL results. Returning empty.")
            return "No relevant information found in the knowledge base."

        # 4. Format Results
        formatted = []
        for record in ranked_docs:
            formatted.append(f"[{record.source_metadata}]\n{record.content}") # Ensure content is string
            
        return "\n\n".join(formatted)