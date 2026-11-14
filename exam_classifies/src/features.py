import re
import numpy as np
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.base import BaseEstimator,TransformerMixin

try:
    PRONOUNCING_DICT = cmudict.dict()
except LookupError:
    nltk.download("cmudict",quite=True)
    PRONOUNCING_DICT = cmudict.dict()
BLOOM_VERBS ={
    1: {"define", "list", "recall", "state", "name", "identify", "label", "mention", "write"},          # Remember
    2: {"explain", "describe", "summarize", "classify", "discuss", "interpret", "distinguish"},          # Understand
    3: {"solve", "calculate", "compute", "demonstrate", "apply", "illustrate", "use", "find"},           # Apply
    4: {"analyze", "compare", "contrast", "differentiate", "examine", "deduce", "derive", "prove"},      # Analyze
    5: {"evaluate", "justify", "critique", "assess", "argue", "defend", "validate", "recommend"},        # Evaluate
    6: {"design", "formulate", "construct", "develop", "propose", "synthesize", "devise", "invent"}
}

def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]","",word.lower())
    if not word:
        return 0
    if word in PRONOUNCING_DICT:
        return max([len([y for y in x if y[-1].isdigit()]) for x in PRONOUNCING_DICT[word]])
    vowels = re.findall(r"[aeiouy]+",word)
    count = len(vowels)
    if word.endswith("e") and count > 1:
        count -=1
    return max(1,count)

class LinguisticBloomExtractor(BaseEstimator,TransformerMixin):
    def fit(self,X,y= None):
        return self
    def transform(self,X):
        features = []
        for text in X:
            text_str = str(text)
            sentences = sent_tokenize(text_str)
            words = [w for w in word_tokenize(text_str.lower()) if w.isalpha()]

            num_sentences = max(1,len(sentences))
            num_words = max(1,len(words))

            avg_sentence_len = num_words / num_sentences

            syllable_counts = [count_syllables(w) for w in words]
            syllable_density = sum(syllable_counts)/num_words
            complex_word_ratio = sum(1 for s in syllable_counts if s>= 3)/num_words

            tier_counts = [0] * 6
            highest_bloom_tier = 0
            for w in words:
                for tier,verb_set in BLOOM_VERBS.items():
                    if w in verb_set:
                        tier_counts[tier-1] += 1
                        if tier > highest_bloom_tier:
                            highest_bloom_tier = tier
            bloom_densitites = [c/num_words for c in tier_counts]

            has_subparts = 1.0 if re.search(r"\([a-d]|[ivx]+\)",text_str.lower()) else 0.0
            math_symbol_count = len(re.findall(r"[=+\-*/^∫∑√]",text_str))

            row = [
                avg_sentence_len,
                syllable_density,
                complex_word_ratio,
                highest_bloom_tier /6.0,
                has_subparts,
                math_symbol_count
            ] + bloom_densitites

            features.append(row)
        return np.array(features, dtype=np.float64)
        
