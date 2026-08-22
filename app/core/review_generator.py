"""
Review generator — orchestrates dice roll + prompt assembly + LLM call.

Each call to roll_persona() returns a PersonaCard that already contains
the service selection (Die 7, correlated with age + travel group).
No separate service picker needed.

v2 changes (post-audit):
  - _build_prompts() now returns the full PersonaCard alongside prompts
    so the generator can inspect metadata for deduplication
  - Added service_window deduplication: prevents the same service type
    from appearing in back-to-back reviews (max 2 of same service in any
    rolling window of 5) — kills the "content marketing matrix" fingerprint
  - Added group_window deduplication: prevents the same travel group from
    appearing more than twice in any rolling window of 5 reviews
  - Both caches are cleared by clear_generation_cache()
"""

from __future__ import annotations

import logging
from collections import deque

from app.core.llm import LLMClient, get_llm_client
from app.core.review_randomizer import PersonaCard, roll_persona, format_persona_for_prompt
from app.prompts.system_prompt import build_system_prompt


logger = logging.getLogger(__name__)

_default_client: LLMClient | None = None

# ── Session-level deduplication caches ────────────────────────────────────────

# Opening dedup: prevents reviews from starting with the same 6 words
_opening_cache: set[str] = set()

# Service window: tracks service_id of last N reviews
# Prevents same service from appearing more than _MAX_SERVICE_REPEATS
# times in a rolling window of _SERVICE_WINDOW_SIZE
_SERVICE_WINDOW_SIZE  = 5
_MAX_SERVICE_REPEATS  = 2
_service_window: deque[str] = deque(maxlen=_SERVICE_WINDOW_SIZE)

# Group window: tracks travel_group label of last N reviews
# Prevents same group type from dominating consecutive reviews
_GROUP_WINDOW_SIZE   = 5
_MAX_GROUP_REPEATS   = 2
_group_window: deque[str] = deque(maxlen=_GROUP_WINDOW_SIZE)


def clear_generation_cache() -> None:
    """Clear all session-level deduplication caches."""
    _opening_cache.clear()
    _service_window.clear()
    _group_window.clear()


def _get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = get_llm_client()
    return _default_client


def _build_prompts() -> tuple[str, str, float, PersonaCard]:
    """Roll all 7 dice + friction + outcome → assemble (system_prompt, user_prompt, temp_offset, card)."""
    card = roll_persona()
    persona_block = format_persona_for_prompt(card)

    system_prompt = build_system_prompt(
        service_context=card.resolved_context_str,
        persona_block=persona_block,
    )

    user_prompt = (
        f"Write one Google Maps review for Gola Holidays "
        f"about their {card.service.service_name} service."
    )

    logger.debug(
        "Rolled seed=%d service=%s age=%s region=%s group=%s personality=%s friction=%s outcome=%s",
        card.seed,
        card.service.service_id,
        card.age.label,
        card.region.label[:20],
        card.travel_group.label[:20],
        card.personality.label,
        card.friction_seed.label if card.friction_seed else "none",
        card.emotional_outcome.label,
    )

    return system_prompt, user_prompt, card.temperature_offset, card


def _is_card_deduplicated(card: PersonaCard) -> bool:
    """
    Return True if this card should be rejected due to session-level deduplication.
    Checks: opening phrase (after generation), service window, and group window.
    Called BEFORE generation to short-circuit obvious duplicates.
    """
    # Service window check
    service_count = list(_service_window).count(card.service.service_id)
    if service_count >= _MAX_SERVICE_REPEATS:
        logger.warning(
            "Service '%s' already appears %d times in last %d reviews. Re-rolling.",
            card.service.service_id, service_count, _SERVICE_WINDOW_SIZE,
        )
        return True

    # Group window check
    group_count = list(_group_window).count(card.travel_group.label)
    if group_count >= _MAX_GROUP_REPEATS:
        logger.warning(
            "Travel group '%s' already appears %d times in last %d reviews. Re-rolling.",
            card.travel_group.label, group_count, _GROUP_WINDOW_SIZE,
        )
        return True

    return False


def _record_card(card: PersonaCard) -> None:
    """Record a successfully used card into the deduplication windows."""
    _service_window.append(card.service.service_id)
    _group_window.append(card.travel_group.label)


async def agenerate_review(client: LLMClient | None = None) -> str:
    """Async: generate one unique review with full 7-dice randomization and session dedup."""
    resolved = client or _get_default_client()
    last_card: PersonaCard | None = None
    last_system: str = ""
    last_user: str = ""
    last_offset: float = 0.0

    for _ in range(5):
        system_prompt, user_prompt, temp_offset, card = _build_prompts()

        # Pre-generation dedup: service + group windows
        if _is_card_deduplicated(card):
            continue

        rev = await resolved.agenerate(system_prompt, user_prompt, temp_offset)

        # Post-generation dedup: opening phrase
        words = rev.lower().split()
        if len(words) >= 6:
            opening = " ".join(words[:6])
            if opening in _opening_cache:
                logger.warning("Duplicate opening detected '%s'. Retrying...", opening)
                continue
            _opening_cache.add(opening)

        _record_card(card)
        return rev

    # Fallback after 5 retries: generate without dedup checks
    logger.warning("Dedup retries exhausted — generating without dedup constraints.")
    system_prompt, user_prompt, temp_offset, card = _build_prompts()
    rev = await resolved.agenerate(system_prompt, user_prompt, temp_offset)
    _record_card(card)
    return rev


def generate_review(client: LLMClient | None = None) -> str:
    """Sync: generate one unique review with full 7-dice randomization and session dedup."""
    resolved = client or _get_default_client()

    for _ in range(5):
        system_prompt, user_prompt, temp_offset, card = _build_prompts()

        # Pre-generation dedup: service + group windows
        if _is_card_deduplicated(card):
            continue

        rev = resolved.generate(system_prompt, user_prompt, temp_offset)

        # Post-generation dedup: opening phrase
        words = rev.lower().split()
        if len(words) >= 6:
            opening = " ".join(words[:6])
            if opening in _opening_cache:
                logger.warning("Duplicate opening detected '%s'. Retrying...", opening)
                continue
            _opening_cache.add(opening)

        _record_card(card)
        return rev

    # Fallback after 5 retries: generate without dedup checks
    logger.warning("Dedup retries exhausted — generating without dedup constraints.")
    system_prompt, user_prompt, temp_offset, card = _build_prompts()
    rev = resolved.generate(system_prompt, user_prompt, temp_offset)
    _record_card(card)
    return rev
