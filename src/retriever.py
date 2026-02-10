from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import normalize


class FAQRetriever:
    """
    TF-IDF retriever over FAQ questions.
    - Fit once on startup
    - Retrieve best match for user query
    """

    def __init__(self, qa_items: List[Dict[str, str]]) -> None:
        if not qa_items:
            raise ValueError("qa_items is empty")

        self.qa_items = qa_items
        self.questions = [normalize(item["question"]) for item in qa_items]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def retrieve(
        self,
        user_question: str,
        threshold: float = 0.30,
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Returns (best_item, score). If score < threshold => (None, score)
        """
        q = normalize(user_question)
        if not q:
            return None, 0.0

        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= threshold:
            return self.qa_items[best_idx], best_score

        return None, best_score
