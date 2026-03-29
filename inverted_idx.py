#!/usr/bin/env python3
"""inverted_idx - Inverted index for full-text search with TF-IDF scoring."""
import sys, math, re
from collections import Counter

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docs = {}
        self.doc_count = 0
    def add(self, doc_id, text):
        tokens = self._tokenize(text)
        self.docs[doc_id] = tokens
        self.doc_count += 1
        seen = set()
        for token in tokens:
            if token not in self.index:
                self.index[token] = {}
            self.index[token][doc_id] = self.index[token].get(doc_id, 0) + 1
            seen.add(token)
    def _tokenize(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())
    def search(self, query):
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = Counter()
        for token in tokens:
            if token not in self.index:
                continue
            df = len(self.index[token])
            idf = math.log(self.doc_count / df) if df > 0 else 0
            for doc_id, tf in self.index[token].items():
                scores[doc_id] += tf * idf
        return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    def boolean_and(self, terms):
        terms = [t.lower() for t in terms]
        if not terms:
            return set()
        result = set(self.index.get(terms[0], {}).keys())
        for t in terms[1:]:
            result &= set(self.index.get(t, {}).keys())
        return result
    def boolean_or(self, terms):
        terms = [t.lower() for t in terms]
        result = set()
        for t in terms:
            result |= set(self.index.get(t, {}).keys())
        return result

def test():
    idx = InvertedIndex()
    idx.add(1, "the quick brown fox")
    idx.add(2, "the lazy brown dog")
    idx.add(3, "quick fox jumps over lazy dog")
    results = idx.search("quick fox")
    assert results[0] in (1, 3)  # most relevant
    assert len(results) >= 2
    # boolean AND
    both = idx.boolean_and(["brown", "fox"])
    assert both == {1}
    # boolean OR
    either = idx.boolean_or(["fox", "dog"])
    assert either == {1, 2, 3}
    # no results
    assert idx.search("zebra") == []
    print("OK: inverted_idx")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print("Usage: inverted_idx.py test")
