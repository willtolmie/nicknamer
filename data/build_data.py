import re
import json
from wordfreq import zipf_frequency

try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception:
    nltk = None
    wn = None

CMU_PATH = "cmudict-0.7b"

BLAND_WORDS = {
    "somebody", "anybody", "nobody", "someone", "anyone", "anything", 
    "everything", "nothing", "something", "person", "people", "other",
    "another", "through", "within", "without", "everywhere"
}

def is_proper_name(word, wordnet_module):
    if not wordnet_module:
        return False
    
    synsets = wordnet_module.synsets(word)
    
    # 1. If WordNet doesn't know the word at all, it's likely a proper name 
    # (or a very obscure word already filtered by your zipf_frequency)
    if not synsets:
        return True
    
    # 2. Check if the word is ONLY categorized as a person/name
    # This prevents filtering "Apple" (the fruit) just because "Apple" is a name.
    all_person = all("noun.person" in s.lexname() for s in synsets)
    if all_person:
        return True
        
    return False

def init_wordnet():
    if not nltk or not wn:
        return None
    try:
        wn.synsets("test")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            return None
    return wn

def is_person_word(word, wordnet_module):
    """Checks if the word refers to a person/entity in WordNet."""
    if not wordnet_module:
        return False
    synsets = wordnet_module.synsets(word, pos=wn.NOUN)
    for ss in synsets:
        # Check if the word is a 'person' or 'agent' in the hierarchy
        lexname = ss.lexname()
        if "noun.person" in lexname or "noun.animal" in lexname:
            return True
    return False

def calculate_nickname_score(word, freq, has_meaning, is_person):
    score = 0
    
    # 1. Frequency Sweet Spot (Interesting words are usually 3.0 - 5.0)
    if 3.2 <= freq <= 5.2:
        score += 5
    elif freq > 5.2: # Too common (like 'somebody')
        score += 1
    
    # 2. Semantic value
    if is_person:
        score += 10
    if has_meaning:
        score += 3
        
    # 3. Aesthetic 'Crunchiness' (Z, X, Q, V, K make for cool nicknames)
    if re.search(r'[zxqvk]', word):
        score += 2
        
    return score

def syllable_count(phonemes):
    return sum(1 for p in phonemes if re.search(r"\d", p))

def rhyme_key(phonemes):
    for i in range(len(phonemes)-1, -1, -1):
        if re.search(r"\d", phonemes[i]):
            return "-".join(phonemes[i:])
    return "-".join(phonemes[-2:])

def first_definition(word, wordnet_module):
    if not wordnet_module:
        return None
    synsets = wordnet_module.synsets(word)
    if not synsets:
        return None
    preferred = [s for s in synsets if s.pos() in {"n", "a", "s"}]
    chosen = preferred[0] if preferred else synsets[0]
    return chosen.definition()

# Main processing logic
words_data = {}
rhymes_data = {}
definitions_data = {}
wordnet_module = init_wordnet()

with open(CMU_PATH, "r", encoding="latin-1") as f:
    for line in f:
        if line.startswith(";") or "  " not in line:
            continue
        parts = line.strip().split("  ")
        word = parts[0].lower()
        
        # Clean word (remove variants like 'sophie(1)')
        word = re.sub(r'\(\d\)', '', word)
        
        if not word.isalpha() or word in BLAND_WORDS:
            continue
        if len(word) < 3:
            continue

        phonemes = parts[1].split()
        freq = zipf_frequency(word, "en")
        
        # We lower the floor slightly to 3.0 to catch "cooler" words
        if freq < 3.0: 
            continue

        if is_proper_name(word, wordnet_module):
            continue

        meaning = first_definition(word, wordnet_module)
        is_p = is_person_word(word, wordnet_module)
        
        # Calculate quality
        quality_score = calculate_nickname_score(word, freq, bool(meaning), is_p)
        
        if quality_score <= 5:
            continue

        syl = syllable_count(phonemes)
        rk = rhyme_key(phonemes)

        words_data[word] = {
            "phonemes": phonemes,
            "syllables": syl,
            "rhyme_key": rk,
            "freq": freq,
            "score": quality_score
        }

        if meaning:
            definitions_data[word] = meaning
        
        rhymes_data.setdefault(rk, []).append(word)

# Sort the rhyme lists by our new Score so the best nicknames come first
for rk in rhymes_data:
    rhymes_data[rk].sort(key=lambda w: words_data[w]['score'], reverse=True)

# Save everything
with open("words.json", "w") as f: json.dump(words_data, f)
with open("rhymes.json", "w") as f: json.dump(rhymes_data, f)
with open("definitions.json", "w") as f: json.dump(definitions_data, f)

print(f"Kept {len(words_data)} interesting words.")