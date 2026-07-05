import hashlib
from typing import Dict, Any, Set, Tuple

class CandidateDeduplicator:
    """Detects and resolves duplicate candidate profiles using primary IDs and composite identity hashing."""

    def __init__(self):
        self.seen_ids: Set[str] = set()
        self.seen_identity_hashes: Dict[str, Tuple[str, float, str]] = {} # hash -> (candidate_id, completeness, last_active_date)
        self.ids_to_drop: Set[str] = set()

    def generate_identity_hash(self, candidate_data: Dict[str, Any]) -> str:
        """Generates a SHA-256 hash using key biographical features for entity resolution."""
        profile = candidate_data.get("profile", {})
        edu_list = candidate_data.get("education", [])
        
        # Extract features for identity
        name = str(profile.get("anonymized_name", "")).strip().lower()
        loc = str(profile.get("location_normalized", profile.get("location", ""))).strip().lower()
        country = str(profile.get("country_normalized", profile.get("country", ""))).strip().lower()
        title = str(profile.get("current_title_normalized", profile.get("current_title", ""))).strip().lower()
        
        edu_inst = ""
        edu_deg = ""
        if edu_list and isinstance(edu_list, list) and len(edu_list) > 0:
            first_edu = edu_list[0]
            if isinstance(first_edu, dict):
                edu_inst = str(first_edu.get("institution", "")).strip().lower()
                edu_deg = str(first_edu.get("degree_normalized", first_edu.get("degree", ""))).strip().lower()

        composite = f"{name}_{loc}_{country}_{title}_{edu_inst}_{edu_deg}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    def track_and_check_duplicate(self, candidate_data: Dict[str, Any]) -> bool:
        """Checks if the candidate is a duplicate.
        
        Returns:
            bool: True if this candidate should be dropped (is a duplicate of a better profile), else False.
        """
        cid = candidate_data.get("candidate_id")
        if not cid:
            return True # Drop record without ID
            
        # 1. Primary Key Check
        if cid in self.seen_ids:
            return True
        self.seen_ids.add(cid)

        # 2. Semantic/Composite Identity Check
        identity_hash = self.generate_identity_hash(candidate_data)
        completeness = float(candidate_data.get("redrob_signals", {}).get("profile_completeness_score", 0.0))
        last_active = str(candidate_data.get("redrob_signals", {}).get("last_active_date", ""))

        if identity_hash in self.seen_identity_hashes:
            prev_cid, prev_completeness, prev_active = self.seen_identity_hashes[identity_hash]
            
            # Decide which one to keep
            if completeness > prev_completeness or (completeness == prev_completeness and last_active > prev_active):
                # Current record is better: mark the previous one to be ignored/dropped from output (if already written, 
                # but in our batching pipeline we do this in-memory before writing, or track IDs to exclude).
                # Since we stream, we add the previous candidate_id to drop lists and update tracking.
                self.ids_to_drop.add(prev_cid)
                self.seen_identity_hashes[identity_hash] = (cid, completeness, last_active)
                return False # Keep current
            else:
                # Previous record is better: drop current
                return True
        else:
            self.seen_identity_hashes[identity_hash] = (cid, completeness, last_active)
            return False
