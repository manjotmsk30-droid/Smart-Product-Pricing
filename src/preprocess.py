"""
Preprocessing: Text cleaning and quantity/brand parsing.
Strictly deterministic - no external data sources.
Follows Document 12 specifications exactly.

CHANGES (safe, repo-wide compatible):
1) TextCleaner: normalize common unicode fractions, collapse repeated separators,
   and strip stray punctuation while preserving spec tokens. Prevent empty/None.
2) QuantityParser: broader unit lexicon (gm, gms, kgs, litres, pcs/pc/piece),
   support comma decimals and “x” variants, add pattern for "10 pcs"/"12 tablets",
   ensure numeric outputs (float) and consistent has_qty, unit flags.
3) BrandExtractor: trim bracketed prefixes/suffixes, drop TM/®/© marks, and
   avoid leading numeric tokens before brand. Keeps return lowercased.

All feature keys and class names are unchanged to avoid breaking other modules.
"""

import re
import unicodedata
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from html import unescape


class TextCleaner:
    """Clean and normalize text fields according to Document 12."""
    
    def __init__(self):
        self.html_pattern = re.compile(r'<[^>]+>')
        # keep-only punctuation set (spec): x / % - ( ) + space
        self._keep = set(list("x/%-() "))
        # map common unicode fractions to decimals for stable quantity parsing
        self._frac_map = {
            "½": "0.5", "¼": "0.25", "¾": "0.75",
            "⅓": "0.333", "⅔": "0.667", "⅛": "0.125"
        }
        
    def clean(self, text: str) -> str:
        """
        Clean text: remove HTML, normalize unicode, lowercase.
        Keep meaningful punctuation: x, /, %, –, -, (, )
        
        Document 12 spec:
        - Strip HTML, punctuation except "x", "/", "%", "–", "-", "( )"
        - Lowercase and normalize unicode
        - Collapse spaces; standardize separators: replace "–|—|:" → "-"
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        
        text = str(text)
        
        # Unescape HTML entities
        text = unescape(text)
        
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Normalize unicode (NFKC as per spec)
        text = unicodedata.normalize('NFKC', text)

        # Replace unicode fractions before other symbol normalizations
        for k, v in self._frac_map.items():
            if k in text:
                text = text.replace(k, v)
        
        # Standardize separators (Document 12: –|—|: → -)
        text = text.replace('—', '-').replace('–', '-').replace(':', '-')
        text = text.replace('×', 'x').replace('*', 'x')
        # Ensure ' x ' variant around numbers is parsable (e.g., "2x500" -> "2 x 500")
        text = re.sub(r'(\d)([xX])(\d)', r'\1 x \3', text)
        
        # Lowercase
        text = text.lower()
        
        # Drop stray punctuation except allowed spec chars
        text = ''.join(ch for ch in text if ch.isalnum() or ch in self._keep)
        # collapse multiple dashes/spaces
        text = re.sub(r'-{2,}', '-', text)
        # Collapse whitespace
        text = ' '.join(text.split())
        
        return text


class QuantityParser:
    """
    Parse Item Pack Quantity (IPQ) and sizes into standardized features.
    Implements exact regex patterns from Document 12 (hardened).
    """
    
    # Unit conversions to base units (Document 12 spec + aliases)
    UNIT_MAP = {
        'kg': 1000,  # to grams
        'kgs': 1000,
        'g': 1,
        'gm': 1,
        'gms': 1,
        'l': 1000,   # to ml
        'liter': 1000,
        'liters': 1000,
        'litre': 1000,
        'litres': 1000,
        'ml': 1,
        'tb': 1,     # tablets to pieces
        'tablet': 1,
        'tablets': 1,
        'capsule': 1,
        'capsules': 1,
        'pc': 1,
        'pcs': 1,
        'piece': 1,
        'pieces': 1,
        'count': 1,
        'bag': 1,
        'bags': 1,
        'sheet': 1,
        'sheets': 1,
    }
    
    def __init__(self):
        # Regex patterns from Document 12 (apply in order; take first hit)
        self.patterns = [
            # Pattern 1: (?P<count>\d+)\s*[x×*]\s*(?P<size>[\d.,]*\.?\d+)\s*(?P<unit>kg|g|...)
            re.compile(
                r'(\d+)\s*[x×*]\s*([\d.,]*\.?\d+)\s*'
                r'(kg|kgs?|g|gm|gms|l|ml|litres?|liters?|pcs?|pc|piece|pieces|tablets?|capsules?|bags?|sheets?)',
                re.IGNORECASE
            ),
            # Pattern 2: (?P<size>[\d.,]*\.?\d+)\s*(?P<unit>kg|g|...)\s*\((?:pack|po|set)\s*of\s*(?P<count>\d+)\)
            re.compile(
                r'([\d.,]*\.?\d+)\s*(kg|kgs?|g|gm|gms|l|ml|litres?|liters?)\s*'
                r'\((?:pack|po|set)\s*of\s*(\d+)\)',
                re.IGNORECASE
            ),
            # Pattern 3: (?:pack|po|set)\s*of\s*(?P<count>\d+) (assume unitless pieces)
            re.compile(
                r'(?:pack|po|set)\s*of\s*(\d+)',
                re.IGNORECASE
            ),
            # Pattern 3b: "10 pcs" / "12 tablets" / "6 pc" (unit only count)
            re.compile(
                r'(\d+)\s*(pcs?|pc|piece|pieces|tablets?|capsules?|bags?|sheets?)',
                re.IGNORECASE
            ),
            # Pattern 4: (?P<size>[\d.,]*\.?\d+)\s*(?P<unit>kg|g|l|ml) (single unit)
            re.compile(
                r'([\d.,]*\.?\d+)\s*(kg|kgs?|g|gm|gms|l|ml|litres?|liters?)',
                re.IGNORECASE
            ),
        ]
    
    def parse(self, text: str) -> Dict[str, float]:
        """
        Parse quantity information from text.
        
        Document 12 Features computed:
        - pack_count = count or 1
        - size_value = numeric size (float)
        - size_unit_id ∈ {g, ml, pcs} (one-hot)
        - total_qty_std = pack_count * size_value_in_base_unit
        - has_qty (0/1), size_per_pack
        
        Guardrails: clamp impossible totals (e.g., > 1e7) to NaN
        
        Returns:
            dict with: pack_count, size_value, size_unit_*, total_qty_std, has_qty, size_per_pack
        """
        features = {
            'pack_count': 1.0,
            'size_value': np.nan,
            'size_unit_g': 0,
            'size_unit_ml': 0,
            'size_unit_pcs': 0,
            'total_qty_std': np.nan,
            'has_qty': 0,
            'size_per_pack': np.nan,
        }
        
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return features
        
        text = str(text).lower()
        # normalize comma decimals like "2,5l" -> "2.5l"
        text = re.sub(r'(\d),(\d)', r'\1.\2', text)
        
        # Try patterns in order (Document 12: apply in order; take first hit)
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                return self._extract_from_match(match, pattern)
        
        return features
    
    def _extract_from_match(self, match, pattern) -> Dict[str, float]:
        """Extract features from regex match."""
        groups = match.groups()
        
        features = {
            'pack_count': 1.0,
            'size_value': np.nan,
            'size_unit_g': 0,
            'size_unit_ml': 0,
            'size_unit_pcs': 0,
            'total_qty_std': np.nan,
            'has_qty': 1,
            'size_per_pack': np.nan,
        }
        
        try:
            # Pattern 1: count x size unit
            if len(groups) == 3 and groups[0].isdigit():
                count = float(groups[0])
                size = float(groups[1].replace(',', '')) if groups[1] else 1.0
                unit = groups[2].lower()
                
                features['pack_count'] = count
                features['size_value'] = size
                
                unit_key = self._normalize_unit(unit)
                multiplier = self.UNIT_MAP.get(unit_key, 1)
                features['total_qty_std'] = count * size * multiplier
                
                if unit_key in ['kg', 'g']:
                    features['size_unit_g'] = 1
                elif unit_key in ['l', 'ml']:
                    features['size_unit_ml'] = 1
                else:
                    features['size_unit_pcs'] = 1
            
            # Pattern 2: size unit (pack of count)
            elif len(groups) == 3 and not groups[0].isdigit():
                size = float(groups[0].replace(',', '')) if groups[0] else 1.0
                unit = groups[1].lower()
                count = float(groups[2])
                
                features['pack_count'] = count
                features['size_value'] = size
                
                unit_key = self._normalize_unit(unit)
                multiplier = self.UNIT_MAP.get(unit_key, 1)
                features['total_qty_std'] = count * size * multiplier
                
                if unit_key in ['kg', 'g']:
                    features['size_unit_g'] = 1
                elif unit_key in ['l', 'ml']:
                    features['size_unit_ml'] = 1
            
            # Pattern 3: pack of count (unitless)
            elif len(groups) == 1:
                count = float(groups[0])
                features['pack_count'] = count
                features['total_qty_std'] = count
                features['size_unit_pcs'] = 1
                features['size_value'] = np.nan
            
            # Pattern 3b: "10 pcs" / "12 tablets"
            elif len(groups) == 2 and groups[0].isdigit():
                count = float(groups[0])
                unit = groups[1].lower()
                features['pack_count'] = count
                features['total_qty_std'] = count * self.UNIT_MAP.get(self._normalize_unit(unit), 1)
                features['size_unit_pcs'] = 1
                features['size_value'] = np.nan
            
            # Pattern 4: single unit
            elif len(groups) == 2:
                size = float(groups[0].replace(',', ''))
                unit = groups[1].lower()
                
                features['size_value'] = size
                
                unit_key = self._normalize_unit(unit)
                multiplier = self.UNIT_MAP.get(unit_key, 1)
                features['total_qty_std'] = size * multiplier
                
                if unit_key in ['kg', 'g']:
                    features['size_unit_g'] = 1
                elif unit_key in ['l', 'ml']:
                    features['size_unit_ml'] = 1
            
            # Guardrails: clamp impossible totals (Document 12: > 1e7 → NaN)
            if features['total_qty_std'] is not None and features['total_qty_std'] > 1e7:
                features['total_qty_std'] = np.nan
            
            # Compute size_per_pack (Document 12 spec)
            if features['pack_count'] > 0 and not np.isnan(features['total_qty_std']):
                features['size_per_pack'] = features['total_qty_std'] / features['pack_count']
                
        except (ValueError, ZeroDivisionError):
            pass
        
        return features
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit strings to base units."""
        unit = unit.lower().strip()
        
        # Map variations to standard units (Document 12 units map)
        if unit in ['kg', 'kgs', 'kilogram', 'kilograms']:
            return 'kg'
        elif unit in ['g', 'gram', 'grams', 'gm', 'gms']:
            return 'g'
        elif unit in ['l', 'liter', 'liters', 'litre', 'litres']:
            return 'l'
        elif unit in ['ml', 'milliliter', 'milliliters', 'millilitre']:
            return 'ml'
        
        return unit


class BrandExtractor:
    """
    Extract brand from text.
    Document 12 Rule: brand = first token(s) before first "-" or "|" or "," 
    if token length 2–20 and alphabetic; else first capitalized token span from original text.
    """
    
    def __init__(self, min_len: int = 2, max_len: int = 20):
        self.min_len = min_len
        self.max_len = max_len
    
    def extract(self, text: str, original_text: str = None) -> str:
        """
        Extract brand name from text.
        
        Args:
            text: Cleaned lowercase text
            original_text: Original text with capitalization
        """
        if pd.isna(text):
            return "unknown"
        
        text = str(text).strip()

        # Remove bracketed junk and trademark markers from original for fallback path
        clean_orig = (original_text or "").replace("®", "").replace("™", "").replace("©", "")
        clean_orig = re.sub(r'\[.*?\]|\(.*?\)', ' ', clean_orig)
        
        # Document 12: Try to extract from before first separator
        for sep in ['-', '|', ',']:
            if sep in text:
                brand = text.split(sep)[0].strip()
                # Drop leading numerics like "2 x colgate" -> "colgate"
                brand = re.sub(r'^\d+\s*x?\s*', '', brand).strip()
                # Check: token length 2–20 and alphabetic
                if self.min_len <= len(brand) <= self.max_len and brand.replace(' ', '').isalpha():
                    return brand
        
        # Document 12: else first capitalized token span from original text
        if clean_orig and not pd.isna(clean_orig):
            words = str(clean_orig).split()
            for word in words:
                if word and word[0].isupper() and self.min_len <= len(word) <= self.max_len:
                    return word.lower()
        
        # Fallback: first word
        first_word = text.split()[0] if text.split() else "unknown"
        if self.min_len <= len(first_word) <= self.max_len:
            return first_word
        
        return "unknown"


if __name__ == "__main__":
    # Test text cleaner
    cleaner = TextCleaner()
    test_text = "<b>Product™ Name</b> – 500ml × 2"
    cleaned = cleaner.clean(test_text)
    print(f"Cleaned: {cleaned}")
    # Expectation: normalized separators and unicode handling
    assert cleaned == "product name - 500ml x 2", "Text cleaning failed"
    
    # Test quantity parser
    parser = QuantityParser()
    test_cases = [
        ("2 x 500ml", {'pack_count': 2.0, 'total_qty_std': 1000.0, 'has_qty': 1}),
        ("500ml (Pack of 6)", {'pack_count': 6.0, 'total_qty_std': 3000.0, 'has_qty': 1}),
        ("Pack of 3", {'pack_count': 3.0, 'total_qty_std': 3.0, 'has_qty': 1}),
        ("2.5kg", {'total_qty_std': 2500.0, 'has_qty': 1}),
        ("10 pcs", {'pack_count': 10.0, 'total_qty_std': 10.0, 'has_qty': 1}),
        ("1,5l", {'total_qty_std': 1500.0, 'has_qty': 1}),  # comma decimal
    ]
    
    for text, expected in test_cases:
        result = parser.parse(text)
        print(f"\n'{text}':")
        print(f"  Pack count: {result['pack_count']}")
        print(f"  Total qty: {result['total_qty_std']}")
        print(f"  Has qty: {result['has_qty']}")
        for key, val in expected.items():
            assert (pd.isna(result[key]) and pd.isna(val)) or abs(result[key] - val) < 0.01, f"Failed for {text}: {key}"
    
    # Test brand extractor
    extractor = BrandExtractor()
    test_brands = [
        ("colgate - total toothpaste", "Colgate - Total Toothpaste", "colgate"),
        ("dove soap bar", "Dove Soap Bar", "dove"),
    ]
    
    for clean_text, orig_text, expected_brand in test_brands:
        brand = extractor.extract(clean_text, orig_text)
        print(f"\nBrand from '{orig_text}': {brand}")
        assert brand == expected_brand, f"Brand extraction failed for {orig_text}"
    
    print("\n✓ All preprocessing tests passed!")
