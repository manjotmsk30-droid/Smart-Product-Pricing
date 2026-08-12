# fe_text.py  — drop-in hardened version

import re
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class TextFeatureExtractor:
    """Extract features from text using TF-IDF + SVD."""
    def __init__(self, 
                 max_features: int = 150000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 5,
                 n_components: int = 300,
                 random_state: int = 2025):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.n_components = n_components
        self.random_state = random_state

        self.tfidf_word = None
        self.tfidf_char = None
        self.svd_word = None
        self.svd_char = None
        self.scaler = None
        self.fitted = False

    @staticmethod
    def _safe_n_components(requested: int, n_feats: int) -> int:
        """Ensure SVD has at least 1 component and < n_feats."""
        if n_feats <= 1:
            return 1
        return max(1, min(requested, n_feats - 1))

    def fit(self, texts: pd.Series):
        texts = texts.fillna("").astype(str)

        # Word-level TF-IDF
        self.tfidf_word = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            analyzer='word',
            strip_accents='unicode',
            lowercase=True,
        )
        tfidf_word_matrix = self.tfidf_word.fit_transform(texts)

        # Character-level TF-IDF
        self.tfidf_char = TfidfVectorizer(
            max_features=max(1, self.max_features // 2),
            ngram_range=(3, 5),
            min_df=self.min_df,
            analyzer='char',
            strip_accents='unicode',
            lowercase=True,
        )
        tfidf_char_matrix = self.tfidf_char.fit_transform(texts)

        # --- SAFE HANDLING when a vectorizer yields 0 features ---
        if tfidf_word_matrix.shape[1] == 0:
            self.svd_word = None
            svd_word_features = np.zeros((texts.shape[0], 0), dtype=np.float32)
        else:
            n_comp_word = self._safe_n_components(self.n_components, tfidf_word_matrix.shape[1])
            self.svd_word = TruncatedSVD(n_components=n_comp_word, random_state=self.random_state)
            svd_word_features = self.svd_word.fit_transform(tfidf_word_matrix).astype(np.float32, copy=False)

        if tfidf_char_matrix.shape[1] == 0:
            self.svd_char = None
            svd_char_features = np.zeros((texts.shape[0], 0), dtype=np.float32)
        else:
            n_comp_char = self._safe_n_components(max(1, self.n_components // 2), tfidf_char_matrix.shape[1])
            self.svd_char = TruncatedSVD(n_components=n_comp_char, random_state=self.random_state)
            svd_char_features = self.svd_char.fit_transform(tfidf_char_matrix).astype(np.float32, copy=False)
# ----------------------------------------------------------


        # SVD with safe component counts
        n_comp_word = self._safe_n_components(self.n_components, tfidf_word_matrix.shape[1])
        n_comp_char = self._safe_n_components(max(1, self.n_components // 2), tfidf_char_matrix.shape[1])

        self.svd_word = TruncatedSVD(n_components=n_comp_word, random_state=self.random_state)
        svd_word_features = self.svd_word.fit_transform(tfidf_word_matrix).astype(np.float32, copy=False)

        self.svd_char = TruncatedSVD(n_components=n_comp_char, random_state=self.random_state)
        svd_char_features = self.svd_char.fit_transform(tfidf_char_matrix).astype(np.float32, copy=False)

        # Standardize SVD features (keep with_mean=False to avoid downstream changes)
        all_features = np.hstack([svd_word_features, svd_char_features]).astype(np.float32, copy=False)
        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(all_features)

        self.fitted = True

        if os.environ.get("FE_TEXT_VERBOSE", "1") == "1":
            print(f"✓ Fitted TF-IDF: {tfidf_word_matrix.shape[1]} word, {tfidf_char_matrix.shape[1]} char")
            print(f"✓ SVD reduced to {n_comp_word} + {n_comp_char} comps")

    def transform(self, texts: pd.Series) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        texts = texts.fillna("").astype(str)

        tfidf_word_matrix = self.tfidf_word.transform(texts)
        tfidf_char_matrix = self.tfidf_char.transform(texts)

        # Safe word SVD
        try:
            svd_word_features = self.svd_word.transform(tfidf_word_matrix).astype(np.float32, copy=False)
        except Exception:
            svd_word_features = np.zeros((texts.shape[0], 0), dtype=np.float32)

        # Safe char SVD
        try:
            svd_char_features = self.svd_char.transform(tfidf_char_matrix).astype(np.float32, copy=False)
        except Exception:
            svd_char_features = np.zeros((texts.shape[0], 0), dtype=np.float32)


        all_features = np.hstack([svd_word_features, svd_char_features]).astype(np.float32, copy=False)
        return self.scaler.transform(all_features).astype(np.float32, copy=False)

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'tfidf_word': self.tfidf_word,
            'tfidf_char': self.tfidf_char,
            'svd_word': self.svd_word,
            'svd_char': self.svd_char,
            'scaler': self.scaler,
            'config': {
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'n_components': self.n_components,
                'random_state': self.random_state,
                'version': '1.1',
            }
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        instance = cls(**{k: v for k, v in data['config'].items() if k in {
            'max_features','ngram_range','min_df','n_components','random_state'
        }})
        instance.tfidf_word = data['tfidf_word']
        instance.tfidf_char = data['tfidf_char']
        instance.svd_word = data['svd_word']
        instance.svd_char = data['svd_char']
        instance.scaler = data['scaler']
        instance.fitted = True
        return instance


class RuleBasedTextFeatures:
    """Extract rule-based features from text."""
    KEYWORDS = {
        'wireless': ['wireless', 'wifi', 'wi-fi'],
        'bluetooth': ['bluetooth', 'bt'],
        'organic': ['organic', 'bio'],
        'sugar_free': ['sugar free', 'sugar-free', 'sugarfree', 'no sugar'],
        'stainless': ['stainless', 'stainless steel'],
        'kids': ['kids', 'children', 'child'],
        'men': ['men', 'male', "men's"],
        'women': ['women', 'female', "women's"],
        '4g': ['4g', 'lte'],
        '5g': ['5g'],
        'hdmi': ['hdmi'],
        'usb_c': ['usb-c', 'usb c', 'type-c', 'type c'],
        'waterproof': ['waterproof', 'water proof', 'water resistant'],
        'leather': ['leather', 'genuine leather'],
        'cotton': ['cotton', '100% cotton'],
        'refill': ['refill', 're-fill'],
        'combo': ['combo', 'pack', 'bundle'],
    }

    def __init__(self):
        self.feature_names = list(self.KEYWORDS.keys())

    @staticmethod
    def _wrap_token(tok: str) -> str:
        # Add word boundaries only for simple alnum tokens to avoid partial matches.
        if re.fullmatch(r"[A-Za-z0-9]+", tok):
            return rf"\b{re.escape(tok)}\b"
        return re.escape(tok)

    def extract(self, texts: pd.Series) -> pd.DataFrame:
        texts = texts.fillna("").astype(str).str.lower()
        features = pd.DataFrame(index=texts.index)
        for feature_name, keywords in self.KEYWORDS.items():
            pattern = '|'.join(self._wrap_token(kw) for kw in keywords)
            features[f'has_{feature_name}'] = texts.str.contains(pattern, regex=True).astype(int)
        return features


class TextStatFeatures:
    """Extract statistical features from text."""
    @staticmethod
    def extract(texts: pd.Series) -> pd.DataFrame:
        texts = texts.fillna("").astype(str)
        features = pd.DataFrame(index=texts.index)
        features['text_len_chars'] = texts.str.len()
        features['text_len_words'] = texts.str.split().str.len()
        features['avg_word_len'] = features['text_len_chars'] / (features['text_len_words'] + 1)
        features['num_digits'] = texts.str.count(r'\d')
        features['num_uppercase'] = texts.str.count(r'[A-Z]')
        features['num_special_chars'] = texts.str.count(r'[^a-zA-Z0-9\s]')
        features['digit_ratio'] = features['num_digits'] / (features['text_len_chars'] + 1)
        features['upper_ratio'] = features['num_uppercase'] / (features['text_len_chars'] + 1)
        features['has_number'] = texts.str.contains(r'\d', regex=True).astype(int)
        features['has_parentheses'] = texts.str.contains(r'[()]', regex=True).astype(int)
        features['has_hyphen'] = texts.str.contains(r'-', regex=True).astype(int)
        return features


def build_text_features(texts: pd.Series,
                        feature_extractor: Optional[TextFeatureExtractor] = None,
                        fit: bool = False) -> Tuple[np.ndarray, pd.DataFrame, TextFeatureExtractor]:
    texts = pd.Series(texts).astype('string').fillna('')
    if feature_extractor is None:
        feature_extractor = TextFeatureExtractor()
    if fit:
        tfidf_svd = feature_extractor.fit_transform(texts)
    else:
        tfidf_svd = feature_extractor.transform(texts)
    rule_extractor = RuleBasedTextFeatures()
    rule_features = rule_extractor.extract(texts)
    stat_features = TextStatFeatures.extract(texts)
    combined_features = pd.concat([rule_features, stat_features], axis=1)
    return tfidf_svd, combined_features, feature_extractor


if __name__ == "__main__":
    sample_texts = pd.Series([
        "Apple iPhone 13 Pro Max - 256GB - Wireless 5G",
        "Organic Cotton T-Shirt - Pack of 3 - Men",
        "Stainless Steel Water Bottle - 500ml - BPA Free",
        "Samsung Galaxy Buds - Bluetooth Wireless Earbuds",
    ])
    tfidf_features, rule_features, extractor = build_text_features(sample_texts, fit=True)
    print(f"TF-IDF SVD shape: {tfidf_features.shape}")
    print(f"Rule features shape: {rule_features.shape}")
    print(rule_features.head())
