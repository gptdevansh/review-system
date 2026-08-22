"""
Generate 30 reviews with 3-key round-robin rate limiting.

API limits per key: 15 RPM / 250k TPM / 500 RPD
Key pool: 3 keys in round-robin (GeminiClient.KeyPool handles this automatically)
Pacing: 1.5s gap between requests → ~40 RPM total, ~13 RPM per key
Each key is called every 3rd request — well under individual 15 RPM limits.
Expected runtime: ~60 seconds for 30 reviews

v2: uses updated system_prompt, service_prompts, review_randomizer, and review_generator
    Output written to thirty_reviews_v2.txt (separate from v1 batch)

Usage:
    source .venv/bin/activate
    python3 generate_30_reviews.py"""

import asyncio
import time

from app.core.review_generator import agenerate_review, clear_generation_cache

OUTPUT_FILE = "thirty_reviews_v2.txt"
TOTAL        = 30
GAP_SECONDS  = 1.5    # 3 keys × ~13 RPM each = ~40 RPM total, safe under 15 RPM/key
RETRY_WAIT   = 10     # seconds to wait on 429 before switching to next key
MAX_RETRIES  = 6


async def generate_all() -> None:
    clear_generation_cache()
    reviews: list[str] = []
    start = time.time()

    print(f"Generating {TOTAL} reviews at ~{60 // GAP_SECONDS} RPM pacing...")
    print(f"Expected time: ~{(TOTAL * GAP_SECONDS) // 60}m {(TOTAL * GAP_SECONDS) % 60}s\n")

    for i in range(1, TOTAL + 1):
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                rev = await agenerate_review()
                if rev and len(rev.strip()) > 15:
                    reviews.append(rev)
                    elapsed = int(time.time() - start)
                    m, s = divmod(elapsed, 60)
                    print(f"[{i:02d}/{TOTAL}] ✓ {m}m{s:02d}s elapsed — {len(rev.split())} words")
                    print(f"   {rev[:120].strip()}{'...' if len(rev) > 120 else ''}")
                    print()
                    
                    # Append as we go
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"=== REVIEW {i:02d} ===\n{rev}\n\n")
                    
                    break
                else:
                    print(f"[{i:02d}] Empty response, retrying...")
                    attempt += 1
                    await asyncio.sleep(5)

            except Exception as e:
                err_str = str(e).lower()
                attempt += 1
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    wait = RETRY_WAIT * attempt  # backoff: 20s, 40s, 60s, 80s
                    print(f"[{i:02d}] Rate limit hit — waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
                    await asyncio.sleep(wait)
                else:
                    print(f"[{i:02d}] Error: {e} — waiting 10s...")
                    await asyncio.sleep(10)

        else:
            print(f"[{i:02d}] ✗ Failed after {MAX_RETRIES} retries — skipping.")

        # Pace between successful requests (not after a retry sleep)
        if i < TOTAL:
            await asyncio.sleep(GAP_SECONDS)

    total_time = int(time.time() - start)
    m, s = divmod(total_time, 60)
    print(f"─────────────────────────────────────")
    print(f"Done! {len(reviews)}/{TOTAL} reviews generated in {m}m{s:02d}s")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(generate_all())
