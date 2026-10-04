WORD_LIST_PATH = "data/word_list.txt"

def load_word_list():
    with open(WORD_LIST_PATH, "r") as f:
        words = [line.strip().upper() for line in f if line.strip()]
    return words  # already ordered by frequency (most common first)

WORDS = load_word_list()

def suggest_words(prefix, top_n=3):
    """
    Rank candidate words by:
    1. Starts with the prefix
    2. Position in the frequency-ordered word list (earlier = more common = ranked higher)
    """
    prefix = prefix.upper()
    if not prefix:
        return []

    matches = [w for w in WORDS if w.startswith(prefix)]
    return matches[:top_n]

if __name__ == "__main__":
    # quick manual test
    test_prefix = input("Enter a prefix to test suggestions: ")
    results = suggest_words(test_prefix)
    print("Suggestions:", results)