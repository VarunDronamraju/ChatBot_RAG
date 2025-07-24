#!/usr/bin/env python
# -*- coding: utf-8 -*-

def expand_query(query):
    query = query.lower()
    expansions = {
        "ai-driven de novo drug discovery": [
            "artificial intelligence drug design",
            "machine learning drug discovery",
            "de novo molecule design",
            "ai drug development"
        ]
    }
    expanded_queries = [query]
    for key, synonyms in expansions.items():
        if key in query:
            expanded_queries.extend(synonyms)
    return expanded_queries