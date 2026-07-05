import os
import pandas as pd
from typing import List, Dict, Any
from src.ranking.exceptions import ValidationError

class SubmissionGenerator:
    """Formats, validates, and serializes the official hackathon submission CSV file."""

    def validate_and_save(self, ranked_pool: List[Dict[str, Any]], output_path: str) -> None:
        """Runs checks on the ranked pool, extracts the Top 100, and exports to CSV.
        
        Args:
            ranked_pool: Fused, hybrid-scored candidates list sorted descending by score.
            output_path: Target CSV output path.
        """
        # Slice to Top 100
        top_100 = ranked_pool[:100]
        
        if len(top_100) < 100:
            raise ValidationError(f"Ranking pool contains only {len(top_100)} candidates. Exactly 100 are required.")
            
        rows = []
        seen_ids = set()
        
        for idx, cand in enumerate(top_100):
            cid = cand.get("candidate_id")
            score = cand.get("final_score")
            reasoning = cand.get("reasoning", "")
            
            # 1. Non-empty candidate ID
            if not cid:
                raise ValidationError(f"Row {idx+1}: Missing candidate ID.")
                
            # 2. No duplicate candidate IDs
            if cid in seen_ids:
                raise ValidationError(f"Duplicate candidate ID '{cid}' detected in rankings.")
            seen_ids.add(cid)
            
            # 3. Valid score ranges and no NaNs
            if score is None or pd.isna(score):
                raise ValidationError(f"Candidate {cid} has NaN final score.")
            score_val = float(score)
            
            # 4. Reasoning word count checks
            words = reasoning.split()
            word_count = len(words)
            if word_count < 10 or word_count > 50:
                raise ValidationError(
                    f"Candidate {cid}: Reasoning word count of {word_count} violates "
                    f"the strict constraint [10, 50] words. Text: '{reasoning}'"
                )
                
            rows.append({
                "candidate_id": cid,
                "rank": idx + 1,
                "score": score_val,
                "reasoning": reasoning
            })
            
        df_sub = pd.DataFrame(rows)
        
        # 5. Unique ranks verification
        if not df_sub["rank"].is_unique:
            raise ValidationError("Generated ranks are not unique.")
            
        # 6. Monotonicity validation (scores sorted descending)
        scores_list = df_sub["score"].tolist()
        is_sorted = all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1))
        if not is_sorted:
            raise ValidationError("Final scores are not sorted in descending order.")
            
        # Write to CSV
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df_sub.to_csv(output_path, index=False)
        print(f"Submission successfully written to {output_path} (Validated: 100 rows, unique ranks, descending scores).")
