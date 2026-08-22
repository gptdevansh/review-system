"""
Deep audit script for 30 generated reviews.
Checks: phrase duplicates, semantic clusters, service distribution,
opening word variety, Hinglish density, Gola Holidays mention rate.
"""
import re
from collections import Counter, defaultdict

with open("thirty_reviews_v2.txt", encoding="utf-8") as f:
    raw = f.read()

# Split into individual reviews
blocks = re.split(r"=== REVIEW \d+ ===\n", raw)
reviews = [b.strip() for b in blocks if b.strip()]
print(f"Total reviews loaded: {len(reviews)}\n")

# ── 1. Opening word analysis ───────────────────────────────────────────────────
print("=" * 60)
print("1. OPENING WORDS (first word of each review)")
print("=" * 60)
first_words = [r.split()[0] for r in reviews]
word_count = Counter(first_words)
for word, count in word_count.most_common():
    bar = "█" * count
    flag = " ⚠️ " if count >= 3 else ""
    print(f"  {word:<20} {count}x  {bar}{flag}")

# ── 2. Opening phrase (first 5 words) ─────────────────────────────────────────
print("\n" + "=" * 60)
print("2. OPENING PHRASES (first 5 words)")
print("=" * 60)
opening_phrases = [" ".join(r.split()[:5]) for r in reviews]
phrase_count = Counter(opening_phrases)
for phrase, count in phrase_count.most_common():
    flag = " ⚠️ DUPLICATE" if count > 1 else ""
    print(f"  [{count}x] {phrase}{flag}")

# ── 3. Exact 5-gram duplicates across ALL reviews ─────────────────────────────
print("\n" + "=" * 60)
print("3. REPEATED 5-GRAMS (exact phrase patterns)")
print("=" * 60)
ngram_to_reviews = defaultdict(list)
for idx, review in enumerate(reviews, 1):
    words = review.lower().split()
    for i in range(len(words) - 4):
        ngram = " ".join(words[i:i+5])
        ngram_to_reviews[ngram].append(idx)

repeated = {ng: idxs for ng, idxs in ngram_to_reviews.items() if len(idxs) >= 2}
# Filter out very generic / noise ngrams
noise = {"gola holidays for our family", "gola holidays for our", "booked the jim corbett"}
repeated_clean = {k: v for k, v in repeated.items() if k not in noise}

if repeated_clean:
    for ngram, idxs in sorted(repeated_clean.items(), key=lambda x: -len(x[1])):
        print(f"  ⚠️  \"{ngram}\" → Reviews {idxs}")
else:
    print("  ✅ No repeated 5-grams found!")

# ── 4. Semantic concept frequency ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. SEMANTIC CONCEPT FREQUENCY (what topics appear)")
print("=" * 60)
concepts = {
    "Driver punctuality": [r"(on time|before time|came early|waited|time at|reached.*time|arrived.*time|came right on time|exactly at the time|time par)", reviews],
    "Hassle-free logistics": [r"(hassle.free|no tension|tension free|no issue|sorted|stress.free|peace of mind|no trouble|no problem)", reviews],
    "Value for money": [r"(paisa vasool|worth|value for money|worth the money|worth every)", reviews],
    "Driver praised": [r"(driver|bhaiy|chauffeur)", reviews],
    "Kids mentioned": [r"(kid|child|bachh|daughter|son|bett[ae])", reviews],
    "Innova Crysta": [r"(innova|crysta)", reviews],
    "Smoothly/properly": [r"(smooth|properly|nicely)", reviews],
    "Tiger mention": [r"(tiger|pugmark)", reviews],
    "Elephant mention": [r"(elephant|tusker|herd)", reviews],
    "Safari (general)": [r"(safari|jeep|canter|jungle)", reviews],
    "Hotel/resort stay": [r"(resort|hotel|room|check.in|stay)", reviews],
    "Cab/taxi": [r"(cab|taxi|innova|tempo|dzire)", reviews],
}

for concept, (pattern, _) in concepts.items():
    matches = [i+1 for i, r in enumerate(reviews) if re.search(pattern, r, re.IGNORECASE)]
    total = len(reviews)
    pct = len(matches) / total * 100
    bar = "█" * len(matches)
    flag = " ⚠️ HIGH" if len(matches) >= total * 0.6 else (" 📊 OK" if len(matches) >= total * 0.25 else "")
    print(f"  {concept:<28} {len(matches):>2}/{total}  ({pct:.0f}%)  {bar}{flag}")

# ── 5. Gola Holidays mention rate ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 'GOLA HOLIDAYS' MENTION RATE")
print("=" * 60)
gola_mentions = [i+1 for i, r in enumerate(reviews) if "gola holidays" in r.lower()]
total = len(reviews)
print(f"  Mentioned in: {len(gola_mentions)}/{total} reviews ({len(gola_mentions)/total*100:.0f}%)")
print(f"  Reviews WITHOUT mention: {[i+1 for i in range(total) if i+1 not in gola_mentions]}")
print(f"  {'⚠️  Too high (every review)' if len(gola_mentions) == total else '✅ Good variation' if len(gola_mentions) < total * 0.80 else '📊 Acceptable'}")

# ── 6. Hinglish density ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. HINGLISH / HINDI WORD DENSITY")
print("=" * 60)
hinglish_words = r"(bhaiy|bachh|accha|badhiya|vasool|thoda|pareshani|aaram|tense|mast|ekdum|shukriya|sahab|ji\b|sab\b|bahut|bilkul|bohot|raha|thi\b|tha\b|nahi|koi\b|bada|chote|kuch|aur\b|par\b|ke\b|ko\b|se\b|mein\b|ne\b|ka\b|ki\b|hota|milta|gaya|phir|toh\b|sirf|wahan|jaana|karna|maza|raat|subah|paisa|zaroori)"
for i, r in enumerate(reviews, 1):
    matches = re.findall(hinglish_words, r, re.IGNORECASE)
    density = len(matches)
    bar = "▪" * min(density, 20)
    label = "🟢 Heavy Hinglish" if density >= 8 else ("🔵 Moderate" if density >= 3 else "⚪ English dominant")
    print(f"  R{i:02d}: {density:>2} words  {bar:<20}  {label}")

# ── 7. Review length distribution ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. REVIEW LENGTH DISTRIBUTION (words)")
print("=" * 60)
lengths = [(i+1, len(r.split())) for i, r in enumerate(reviews)]
buckets = {"one-liner (≤20)": [], "short (21-45)": [], "medium (46-90)": [], "long (91+)": []}
for idx, wc in lengths:
    if wc <= 20: buckets["one-liner (≤20)"].append(idx)
    elif wc <= 45: buckets["short (21-45)"].append(idx)
    elif wc <= 90: buckets["medium (46-90)"].append(idx)
    else: buckets["long (91+)"].append(idx)

for label, idxs in buckets.items():
    bar = "█" * len(idxs)
    print(f"  {label:<22} {len(idxs):>2}x  {bar}  {idxs}")

# ── 8. Closing word patterns ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. CLOSING PATTERNS (last 4 words of each review)")
print("=" * 60)
closings = [" ".join(r.split()[-4:]).lower() for r in reviews]
closing_count = Counter(closings)
for closing, count in closing_count.most_common(10):
    flag = " ⚠️" if count > 1 else ""
    print(f"  [{count}x] ...{closing}{flag}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
