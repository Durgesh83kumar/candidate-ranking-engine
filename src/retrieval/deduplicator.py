from typing import List, Dict, Any
from src.retrieval.exceptions import DeduplicationError

class CandidateDeduplicator:
    """Ensures each candidate is represented once. Merges match types, queries, and evidence."""

    def deduplicate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates candidates by ID, merging queries and evidence fields.
        
        Args:
            candidates: Rich dictionary records.
            
        Returns:
            List[Dict[str, Any]]: Unique candidate records.
        """
        seen = {}
        
        try:
            for cand in candidates:
                cid = cand.get("candidate_id")
                if not cid:
                    continue
                    
                if cid not in seen:
                    # Make a copy to avoid mutating inputs
                    seen[cid] = dict(cand)
                else:
                    existing = seen[cid]
                    
                    # 1. Keep highest RRF score
                    if cand.get("rrf_score", 0.0) > existing.get("rrf_score", 0.0):
                        # Update scores, titles, and basic stats
                        existing["rrf_score"] = cand["rrf_score"]
                        existing["current_title"] = cand.get("current_title", existing.get("current_title"))
                        existing["current_company"] = cand.get("current_company", existing.get("current_company"))
                        
                    # 2. Merge matched queries
                    merged_queries = list(set(existing.get("matched_queries", []) + cand.get("matched_queries", [])))
                    existing["matched_queries"] = sorted(merged_queries)
                    
                    # 3. Merge matched profile sections
                    merged_evidence = list(set(existing.get("matched_profile_sections", []) + cand.get("matched_profile_sections", [])))
                    existing["matched_profile_sections"] = sorted(merged_evidence)
                    
                    # 4. Merge query similarity listings
                    existing_sims = existing.get("query_similarities", {})
                    new_sims = cand.get("query_similarities", {})
                    existing_sims.update(new_sims)
                    existing["query_similarities"] = existing_sims
                    
                    # 5. Merge query rank listings (keep highest rank/lowest index value)
                    existing_ranks = existing.get("query_ranks", {})
                    new_ranks = cand.get("query_ranks", {})
                    for q, r in new_ranks.items():
                        if q in existing_ranks:
                            existing_ranks[q] = min(existing_ranks[q], r)
                        else:
                            existing_ranks[q] = r
                    existing["query_ranks"] = existing_ranks
                    
            return list(seen.values())
            
        except Exception as e:
            raise DeduplicationError(f"Deduplication process failed: {str(e)}") from e
