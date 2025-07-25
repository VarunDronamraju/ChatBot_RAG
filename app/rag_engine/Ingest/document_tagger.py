# app/rag_engine/Ingest/document_tagger.py

import re
import json
from typing import List, Dict, Set, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer
import spacy

class DocumentTagger:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        # Initialize tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load spaCy model (optional, fallback to NLTK if not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            print("[WARNING] spaCy model not found. Using NLTK only. Install with: python -m spacy download en_core_web_sm")
            self.use_spacy = False
    
    def extract_keywords_and_tags(self, text: str, filename: str, max_keywords: int = 15) -> Dict[str, any]:
        """
        Extract keywords, tags, entities, and document type from text
        Returns: {
            'keywords': List[str],
            'tags': List[str], 
            'entities': List[str],
            'doc_type': str,
            'summary_phrases': List[str]
        }
        """
        result = {
            'keywords': [],
            'tags': [],
            'entities': [],
            'doc_type': 'document',
            'summary_phrases': []
        }
        
        # Clean and preprocess text
        clean_text = self._clean_text(text)
        
        # Extract entities (people, organizations, locations)
        entities = self._extract_entities(clean_text)
        result['entities'] = entities
        
        # Extract keywords using TF-IDF-like approach
        keywords = self._extract_keywords(clean_text, max_keywords)
        result['keywords'] = keywords
        
        # Detect document type based on content patterns
        doc_type = self._detect_document_type(clean_text, filename, entities, keywords)
        result['doc_type'] = doc_type
        
        # Generate contextual tags
        tags = self._generate_contextual_tags(clean_text, entities, keywords, doc_type)
        result['tags'] = tags
        
        # Extract key phrases for better matching
        summary_phrases = self._extract_key_phrases(clean_text)
        result['summary_phrases'] = summary_phrases
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,\-\']', ' ', text)
        return text.strip()
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (people, organizations, locations)"""
        entities = set()
        
        if self.use_spacy:
            # Use spaCy for better entity recognition
            doc = self.nlp(text[:1000000])  # Limit text length for performance
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    if len(ent.text.strip()) > 2:
                        entities.add(ent.text.strip())
        else:
            # Fallback to NLTK
            sentences = sent_tokenize(text[:50000])  # Limit for performance
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity = ' '.join([token for token, pos in chunk.leaves()])
                        if len(entity.strip()) > 2:
                            entities.add(entity.strip())
        
        return list(entities)[:10]  # Limit to top 10 entities
    
    def _extract_keywords(self, text: str, max_keywords: int) -> List[str]:
        """Extract keywords using frequency and importance scoring"""
        # Tokenize and filter
        tokens = word_tokenize(text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if (len(token) > 2 and 
                token.isalpha() and 
                token not in self.stop_words and
                not token.isdigit()):
                filtered_tokens.append(self.lemmatizer.lemmatize(token))
        
        # Count frequency
        token_freq = Counter(filtered_tokens)
        
        # Get POS tags to prioritize nouns and adjectives
        if self.use_spacy:
            doc = self.nlp(text[:500000])
            important_tokens = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    len(token.text) > 2 and 
                    token.text.lower() not in self.stop_words):
                    important_tokens.append(token.lemma_.lower())
        else:
            sentences = sent_tokenize(text)
            important_tokens = []
            for sentence in sentences[:50]:  # Limit sentences for performance
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                for token, pos in pos_tags:
                    if (pos.startswith('NN') or pos.startswith('JJ')) and len(token) > 2:
                        important_tokens.append(self.lemmatizer.lemmatize(token.lower()))
        
        # Score tokens (frequency + importance)
        scored_tokens = {}
        for token in set(important_tokens):
            freq_score = token_freq.get(token, 0)
            importance_score = important_tokens.count(token)
            scored_tokens[token] = freq_score + importance_score * 2
        
        # Get top keywords
        top_keywords = sorted(scored_tokens.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, score in top_keywords[:max_keywords]]
    
    def _detect_document_type(self, text: str, filename: str, entities: List[str], keywords: List[str]) -> str:
        """Detect document type using content analysis"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Create feature vectors for classification
        features = {
            'has_personal_pronouns': bool(re.search(r'\b(i|my|me|myself)\b', text_lower)),
            'has_business_terms': any(term in text_lower for term in ['company', 'business', 'corporation', 'enterprise', 'organization']),
            'has_technical_terms': any(term in text_lower for term in ['algorithm', 'software', 'development', 'programming', 'technology']),
            'has_formal_language': bool(re.search(r'\b(hereby|whereas|pursuant|therefore)\b', text_lower)),
            'has_narrative_markers': any(term in text_lower for term in ['once upon', 'long ago', 'meanwhile', 'suddenly', 'chapter']),
            'has_contact_info': bool(re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)),
            'has_dates': bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)),
            'person_entities': len([e for e in entities if any(name_part.istitle() for name_part in e.split())]),
            'org_entities': len([e for e in entities if any(term in e.lower() for term in ['inc', 'ltd', 'corp', 'company'])]),
        }
        
        # Rule-based classification
        if features['has_contact_info'] and features['has_personal_pronouns']:
            if any(term in filename_lower for term in ['cv', 'resume']):
                return 'resume'
            elif any(term in filename_lower for term in ['cover', 'letter']):
                return 'cover_letter'
        
        if features['has_narrative_markers'] or any(term in keywords for term in ['story', 'tale', 'adventure']):
            return 'story'
        
        if features['person_entities'] > 2 and any(term in keywords for term in ['born', 'life', 'career', 'biography']):
            return 'biography'
        
        if features['has_business_terms'] and features['org_entities'] > 0:
            if any(term in keywords for term in ['ceo', 'executive', 'president', 'director']):
                return 'executive_profile'
            else:
                return 'business_document'
        
        if features['has_technical_terms']:
            return 'technical_document'
        
        if features['has_formal_language']:
            return 'legal_document'
        
        return 'general_document'
    
    def _generate_contextual_tags(self, text: str, entities: List[str], keywords: List[str], doc_type: str) -> List[str]:
        """Generate contextual tags based on document analysis"""
        tags = set()
        
        # Add document type as primary tag
        tags.add(doc_type)
        
        # Add entity-based tags
        for entity in entities:
            if len(entity.split()) == 1:  # Single word entities become tags
                tags.add(entity.lower())
            else:  # Multi-word entities get abbreviated
                words = entity.split()
                if len(words) == 2:
                    tags.add(f"{words[0].lower()}_{words[1].lower()}")
        
        # Add top keywords as tags
        for keyword in keywords[:8]:  # Top 8 keywords
            tags.add(keyword)
        
        # Add contextual tags based on content patterns
        text_lower = text.lower()
        
        # Industry/domain tags
        if any(term in text_lower for term in ['artificial intelligence', 'machine learning', 'ai', 'ml']):
            tags.add('artificial_intelligence')
        if any(term in text_lower for term in ['finance', 'banking', 'investment', 'financial']):
            tags.add('finance')
        if any(term in text_lower for term in ['healthcare', 'medical', 'doctor', 'patient']):
            tags.add('healthcare')
        if any(term in text_lower for term in ['education', 'learning', 'teaching', 'academic']):
            tags.add('education')
        
        # Content type tags
        if any(term in text_lower for term in ['research', 'study', 'analysis', 'findings']):
            tags.add('research')
        if any(term in text_lower for term in ['tutorial', 'guide', 'how to', 'steps']):
            tags.add('instructional')
        if any(term in text_lower for term in ['news', 'report', 'announcement', 'press']):
            tags.add('news')
        
        return list(tags)[:12]  # Limit to 12 tags
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases for better semantic matching"""
        sentences = sent_tokenize(text)
        key_phrases = []
        
        # Extract noun phrases and important sentences
        for sentence in sentences[:20]:  # Limit for performance
            if len(sentence.split()) > 5 and len(sentence.split()) < 30:
                # Look for sentences with important keywords
                if any(marker in sentence.lower() for marker in ['ceo', 'president', 'director', 'founded', 'born', 'known for']):
                    key_phrases.append(sentence.strip())
        
        return key_phrases[:5]  # Top 5 key phrases
    
    def query_intent_analysis(self, query: str) -> Dict[str, any]:
        """Analyze user query to determine search intent and relevant tags"""
        query_lower = query.lower()
        
        # Extract entities from query
        query_entities = self._extract_entities(query)
        
        # Extract keywords from query
        query_keywords = self._extract_keywords(query, max_keywords=5)
        
        # Determine query type
        query_type = 'general'
        if any(word in query_lower for word in ['who is', 'what is', 'tell me about']):
            query_type = 'factual'
        elif any(word in query_lower for word in ['how to', 'explain', 'describe']):
            query_type = 'instructional'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            query_type = 'comparison'
        
        return {
            'entities': query_entities,
            'keywords': query_keywords,
            'query_type': query_type,
            'search_tags': query_entities + query_keywords
        }
