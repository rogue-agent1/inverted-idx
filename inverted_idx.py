#!/usr/bin/env python3
"""inverted_idx - Inverted index for full-text search with TF-IDF ranking."""
import sys, math, re
from collections import defaultdict, Counter

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)  # term -> {doc_id: count}
        self.doc_lengths = {}
        self.doc_count = 0
    def _tokenize(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())
    def add(self, doc_id, text):
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        for term, count in counts.items():
            self.index[term][doc_id] = count
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1
    def search(self, query, top_k=10):
        terms = self._tokenize(query)
        scores = defaultdict(float)
        for term in terms:
            if term not in self.index: continue
            df = len(self.index[term])
            idf = math.log(self.doc_count / df) if df > 0 else 0
            for doc_id, tf in self.index[term].items():
                scores[doc_id] += tf * idf
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

def test():
    idx = InvertedIndex()
    idx.add(1, "the quick brown fox jumps over the lazy dog")
    idx.add(2, "the quick brown cat sits on the mat")
    idx.add(3, "a fox and a dog are friends")
    results = idx.search("fox dog")
    assert results[0][0] == 1  # doc 1 has both fox and dog
    assert len(results) >= 2
    r2 = idx.search("cat mat")
    assert r2[0][0] == 2
    r3 = idx.search("nonexistent")
    assert len(r3) == 0
    print("inverted_idx: all tests passed")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("Usage: inverted_idx.py --test")
