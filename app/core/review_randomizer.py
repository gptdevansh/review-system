"""
Review Randomizer — 7-Dice Generative Persona Engine.

Design philosophy:
  Old system: fixed pool of 12 archetypes + word prescription lists
    → LLM cycles through the same phrases across reviews

  New system: 7 independent dice compose a unique person every call
    → Thousands of combinations, zero word prescriptions
    → LLM infers natural language FROM the character — not from a vocabulary menu

Dice:
  1. Age bucket         — energy level, vocabulary complexity, tech-savviness
  2. Region/Language    — sentence structure, Hinglish ratio, cultural tone
  3. Travel group       — what they notice, whose comfort matters, group energy
  4. Personality type   — emotional register and narrative style
  5. Typing style       — grammar level, sentence rhythm, punctuation habits
  6. Priority           — which aspect of the trip they focus on most
  7. Service used       — correlated with Die 1 (age) + Die 3 (group), NOT pure random

Optional Rolls:
  Friction Seed         — 25% chance: one minor real-world imperfection is injected
  Emotional Outcome     — always rolled: controls the emotional landing of the review
                          (replaces the default "peace/relief/unforgettable" monoculture)

v2 changes (post-audit):
  - Added FrictionSeed die (25% probability): kills the "zero imperfection" AI tell
  - Added EmotionalOutcome die: breaks the "peace of mind / unforgettable" monoculture
  - Added post-roll anxiety-arc override: prevents "senior family + logistics" combo
    from generating the guaranteed problem→solution narrative structure
  - PersonaCard updated to carry friction_seed and emotional_outcome
  - format_persona_for_prompt() injects both into the character block

Possible combinations: 5 × 5 × 5 × 5 × 5 × 5 × 8 outcomes × 2 (friction/no friction)
= ~125,000 base combinations (before length and temperature variation)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from app.prompts.service_prompts import ALL_SERVICES, ServiceContext


# ── Length profiles ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LengthProfile:
    label: str
    sentence_range: tuple[int, int]
    word_hint: str
    description: str


_LENGTH_PROFILES: list[LengthProfile] = [
    LengthProfile(
        label="one-liner",
        sentence_range=(1, 1),
        word_hint="8 to 20 words max",
        description="Ultra short, punchy — quick tap-and-go Google review",
    ),
    LengthProfile(
        label="short",
        sentence_range=(1, 2),
        word_hint="15 to 40 words",
        description="Brief but contains one specific detail, no filler",
    ),
    LengthProfile(
        label="medium",
        sentence_range=(3, 4),
        word_hint="50 to 90 words",
        description="Standard review — a couple of specific details, natural close",
    ),
    LengthProfile(
        label="long",
        sentence_range=(5, 7),
        word_hint="100 to 170 words",
        description="Detailed and textured — sets a scene, has a sense of the trip",
    ),
]

# Mirrors real review length distribution — short reviews are more common
_LENGTH_WEIGHTS: list[float] = [0.12, 0.30, 0.38, 0.20]


# ── Friction Seed (Optional Die — 25% probability) ────────────────────────────
#
# Injects one minor, harmless real-world imperfection into 1 in 4 reviews.
# This is the single most powerful human-authenticity signal:
#   → AI never includes mild negatives in a 5-star review unless forced
#   → Real customers almost always mention at least one small thing
#   → Total absence of imperfection across a batch is a stronger AI signal
#     than any single review's wording

@dataclass(frozen=True)
class FrictionSeed:
    label: str
    description: str  # Injected into persona block; LLM weaves it in naturally


_FRICTION_SEEDS: list[FrictionSeed] = [
    FrictionSeed(
        label="late-start",
        description=(
            "The vehicle or guide arrived 15-20 minutes later than the agreed time. "
            "You weren't particularly annoyed — it happens on mountain roads — but "
            "mention it matter-of-factly. Do not apologize for it or dismiss it. "
            "Just note it as something that happened."
        ),
    ),
    FrictionSeed(
        label="no-wifi",
        description=(
            "The hotel or resort had poor or non-existent wifi. You noticed it. "
            "You either didn't mind (wanted a break from screens anyway) "
            "or mention it as a passing note — not a deal-breaker, just a fact."
        ),
    ),
    FrictionSeed(
        label="bumpy-road",
        description=(
            "One stretch of the mountain road was rough, potholed, or under repair — "
            "quite bumpy for 15-20 minutes. The vehicle handled it fine but you felt it. "
            "Mention it as a passing observation without dramatizing."
        ),
    ),
    FrictionSeed(
        label="no-big-sighting",
        description=(
            "The safari did not produce the big wildlife sighting you were hoping for — "
            "no tiger, or the tiger was spotted only briefly and very far away. "
            "You still had a good overall time. Acknowledge the miss plainly, "
            "without forcing fake optimism about it."
        ),
    ),
    FrictionSeed(
        label="crowded-spot",
        description=(
            "One part of the trip — a viewpoint, a ghat, a temple, or a lake — was "
            "more crowded than you had expected. You found a way around it or accepted it. "
            "Mention it as context, not as a serious complaint."
        ),
    ),
    FrictionSeed(
        label="check-in-wait",
        description=(
            "Check-in at the hotel or resort took longer than expected — maybe 20-30 minutes "
            "of waiting even though the booking was pre-confirmed. You got your room and it "
            "was fine. Mention the wait briefly as part of the experience."
        ),
    ),
    FrictionSeed(
        label="room-smaller",
        description=(
            "The room was slightly smaller than the photos suggested — not cramped, "
            "but noticeably compact. You barely spent time inside anyway, so it didn't matter. "
            "Worth a passing note."
        ),
    ),
    FrictionSeed(
        label="food-miss",
        description=(
            "The food at the hotel or a local dhaba on the way was average — not bad, "
            "but not memorable either. You found something better nearby or just lived with it. "
            "Mention it briefly as a minor observation."
        ),
    ),
]

# 25% of reviews get a friction seed
_FRICTION_PROBABILITY: float = 0.25


# ── Emotional Outcome Die ──────────────────────────────────────────────────────
#
# Controls the emotional DESTINATION of the review — what feeling the final
# sentence or observation lands on.
#
# Without this die, the system defaults every review to:
#   "peace of mind / stress-free / truly unforgettable / relief"
# — which is the #2 most common AI-batch fingerprint (after the anxiety arc).
# This die forces real emotional variety across the batch.

@dataclass(frozen=True)
class EmotionalOutcome:
    label: str
    landing_note: str  # The emotional color the review should end on


_EMOTIONAL_OUTCOMES: list[EmotionalOutcome] = [
    EmotionalOutcome(
        label="quiet satisfaction",
        landing_note=(
            "The trip simply worked. No peak emotional moment — just a clean, satisfying "
            "experience that delivered what it promised. The review ends on calm approval, "
            "not euphoria. The tone is matter-of-fact: 'it was good, that's it.'"
        ),
    ),
    EmotionalOutcome(
        label="still buzzing",
        landing_note=(
            "You're still riding the energy of the trip. Sentences are shorter and punchier. "
            "The excitement is genuine and comes through in rhythm, not superlatives. "
            "The review ends on forward-leaning energy — the feeling hasn't settled yet."
        ),
    ),
    EmotionalOutcome(
        label="childlike delight",
        landing_note=(
            "Something specific — a child's reaction, an unexpected animal, a view — produced "
            "a moment of pure, uncomplicated happiness. The review ends on that one moment "
            "of delight, described through behavior or observation rather than adjectives."
        ),
    ),
    EmotionalOutcome(
        label="awe at the place itself",
        landing_note=(
            "The natural environment — the forest, the peaks, the river, the mist, the silence — "
            "genuinely stopped you. The review ends on the PLACE, not the service. "
            "The landscape is the hero of the last sentence, not the company."
        ),
    ),
    EmotionalOutcome(
        label="ease and rest",
        landing_note=(
            "The trip gave you actual rest — you switched off, got good sleep, felt unhurried. "
            "Express this as a POSITIVE you received ('we actually slept well', 'the mornings "
            "were completely quiet') NOT as the absence of a negative. "
            "Never say 'no stress' — say what the positive experience felt like."
        ),
    ),
    EmotionalOutcome(
        label="nostalgia and longing",
        landing_note=(
            "You already miss the place. The review has a slightly wistful quality — "
            "you describe one thing about the location that you keep thinking about since "
            "returning. It ends on the memory, not on a service verdict."
        ),
    ),
    EmotionalOutcome(
        label="pride in the achievement",
        landing_note=(
            "A sense of accomplishment — especially relevant for harder trips: Kedarnath, "
            "Valley of Flowers, a long mountain drive, a remote homestay. "
            "The review reflects 'we did this and it was worth it' energy — "
            "the satisfaction of having pushed to do something difficult."
        ),
    ),
    EmotionalOutcome(
        label="genuine surprise",
        landing_note=(
            "Something exceeded your expectations — but NOT because you set up a pre-trip worry. "
            "The surprise just happened naturally. You didn't expect the room to have that view. "
            "You didn't expect the forest to feel that dense. "
            "Express genuine mild surprise WITHOUT the manufactured anxiety-arc setup."
        ),
    ),
]


# ── Die 1: Age Bucket ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgeBucket:
    label: str
    character_note: str  # behavioral texture, NOT word instructions


_AGE_BUCKETS: list[AgeBucket] = [
    AgeBucket(
        label="20s",
        character_note=(
            "Young, spontaneous, grew up on the internet. Comfort with short, "
            "punchy writing. Energy comes through naturally. Likely wrote this on a phone."
        ),
    ),
    AgeBucket(
        label="30s",
        character_note=(
            "Working professional or young parent. Values efficiency and clarity. "
            "Writing mixes casual warmth with practical observation."
        ),
    ),
    AgeBucket(
        label="40s",
        character_note=(
            "Mid-life, established, family-oriented or senior professional. "
            "Writes with some deliberateness. Neither too casual nor overly formal."
        ),
    ),
    AgeBucket(
        label="50s",
        character_note=(
            "Nearing or at senior stage. More thoughtful and measured. "
            "Appreciates reliability, courtesy, and things done properly. "
            "Writing style tends to be careful and complete."
        ),
    ),
    AgeBucket(
        label="60s+",
        character_note=(
            "Retired or senior. Takes time with words. Notices the small things — "
            "a courteous driver, a beautiful view, a peaceful moment. "
            "May write longer sentences. Appreciates nature and sincerity."
        ),
    ),
]

_AGE_WEIGHTS: list[float] = [0.20, 0.28, 0.22, 0.18, 0.12]


# ── Die 2: Region / Language Background ───────────────────────────────────────

@dataclass(frozen=True)
class Region:
    label: str
    language_note: str  # how this shapes writing texture — not word lists


_REGIONS: list[Region] = [
    Region(
        label="Metro city (Delhi / Mumbai / Bangalore / Hyderabad)",
        language_note=(
            "Comfortable in English. May use modern casual phrasing. "
            "Urban, direct. Hinglish is possible but not heavy — only where it feels natural."
        ),
    ),
    Region(
        label="Hindi-belt city or town (UP / MP / Rajasthan / Uttarakhand / Bihar)",
        language_note=(
            "Natural Hindi-English mix in daily speech. Sentence structure influenced by Hindi. "
            "Polite and respectful tone is cultural default. "
            "Writes as they would speak — warm, sometimes formal in small ways."
        ),
    ),
    Region(
        label="South India (Tamil Nadu / Karnataka / Kerala / Andhra Pradesh)",
        language_note=(
            "Writes in structured, grammatically careful English. "
            "Minimal Hinglish — only if this person would genuinely know the phrase. "
            "Formal and thoughtful. Complete sentences preferred."
        ),
    ),
    Region(
        label="East India (West Bengal / Odisha / Northeast)",
        language_note=(
            "Slightly descriptive and warm in style. Notices atmosphere and aesthetics. "
            "Gentle personal tone. May frame the review as a small story."
        ),
    ),
    Region(
        label="NRI / living abroad",
        language_note=(
            "Fluent English, slightly more formal. May naturally compare to experiences abroad. "
            "Impressed by local knowledge and authenticity. "
            "Review may reflect both outsider wonder and insider Indian roots."
        ),
    ),
]

_REGION_WEIGHTS: list[float] = [0.30, 0.30, 0.18, 0.12, 0.10]


# ── Die 3: Travel Group ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TravelGroup:
    label: str
    perspective_note: str  # what they observe and care about as a group type


_TRAVEL_GROUPS: list[TravelGroup] = [
    TravelGroup(
        label="Solo traveler",
        perspective_note=(
            "Writes from a personal, first-person singular perspective. "
            "Notices their own individual experience — freedom, value, ease of solo travel."
        ),
    ),
    TravelGroup(
        label="Couple (anniversary or leisure trip)",
        perspective_note=(
            "Writes as 'we'. Notices shared moments, ambiance, and the experience together. "
            "May refer to a partner. Emotional resonance matters."
        ),
    ),
    TravelGroup(
        label="Family with young children",
        perspective_note=(
            "Very aware of the children's experience — their reactions, what they noticed, "
            "what made them excited. Writes from the parent's point of view watching the kids. "
            "Focus on what the children DID and RESPONDED to — not on anxieties managed."
        ),
    ),
    TravelGroup(
        label="Friends group trip",
        perspective_note=(
            "Group energy — writes as 'we all', references shared fun. "
            "Casual and upbeat. The collective good time is the measure of success."
        ),
    ),
    TravelGroup(
        label="Traveling with parents or senior family members",
        perspective_note=(
            "Writes about what the elders NOTICED and ENJOYED — a particular view they loved, "
            "a moment of ease, something that made them smile. "
            "Focus on the elders' positive experience — not on logistics managed or anxieties resolved."
        ),
    ),
]

_TRAVEL_GROUP_WEIGHTS: list[float] = [0.15, 0.22, 0.25, 0.20, 0.18]


# ── Die 4: Personality Type ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Personality:
    label: str
    description: str


_PERSONALITIES: list[Personality] = [
    Personality(
        label="Enthusiastic",
        description=(
            "Genuinely excited. High energy and expressive. Positive emotion comes through "
            "naturally without feeling forced. Exclamations feel earned."
        ),
    ),
    Personality(
        label="Calm and practical",
        description=(
            "Matter-of-fact. States what happened, whether it worked, and gives a verdict. "
            "No drama, no hyperbole. Trust is built through understatement."
        ),
    ),
    Personality(
        label="Grateful and warm",
        description=(
            "Genuinely appreciative. Acknowledges a specific moment or detail in a human way. "
            "Warmth without being gushing or performative."
        ),
    ),
    Personality(
        label="Pleasantly surprised (slight skeptic)",
        description=(
            "Something specifically exceeded what they expected — NOT because they set up "
            "a pre-trip anxiety. The surprise is concrete and specific. "
            "Slightly understated but genuinely positive."
        ),
    ),
    Personality(
        label="Storyteller",
        description=(
            "Sets a small scene or describes a moment before landing the observation. "
            "Review reads like they're telling a friend one specific thing about the trip. "
            "Builds to a single point — not a checklist."
        ),
    ),
]


# ── Die 5: Typing Style ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TypingStyle:
    label: str
    description: str


_TYPING_STYLES: list[TypingStyle] = [
    TypingStyle(
        label="Fast phone typer",
        description=(
            "Short bursts of thought. Doesn't overthink punctuation or grammar. "
            "May skip commas. Occasionally runs sentences together. Direct and unpolished."
        ),
    ),
    TypingStyle(
        label="Careful and structured writer",
        description=(
            "Full, complete sentences. Proper punctuation. Reads like they thought about "
            "what to say before typing. Consistent and clear."
        ),
    ),
    TypingStyle(
        label="Natural Hinglish code-switcher",
        description=(
            "Moves between Hindi and English mid-thought — not forced, just how they think. "
            "The Hindi words emerge where they feel more natural than the English equivalent. "
            "The sentence structure itself is sometimes Hindi-influenced — not just English "
            "with Hindi words inserted. Fragments are grammatically correct in Hindi logic."
        ),
    ),
    TypingStyle(
        label="Gen-Z internet style",
        description=(
            "Very brief. Minimal punctuation. May use casual internet abbreviations "
            "where they'd genuinely use them. Maximum meaning with minimum words."
        ),
    ),
    TypingStyle(
        label="Formal and measured",
        description=(
            "Polite, complete language. Would not use slang. "
            "Reads like a considered feedback response, not a casual tap."
        ),
    ),
]


# ── Die 6: What They Prioritize ───────────────────────────────────────────────

@dataclass(frozen=True)
class Priority:
    label: str
    focus_note: str  # what aspect of the trip their review naturally gravitates toward


_PRIORITIES: list[Priority] = [
    Priority(
        label="Value for money",
        focus_note=(
            "Their review gravitates toward whether the trip was worth the cost. "
            "Not necessarily cheap — just fair and worthwhile."
        ),
    ),
    Priority(
        label="Physical comfort and convenience",
        focus_note=(
            "Notices quality of vehicle, hotel room, road conditions, ease of travel. "
            "The physical experience of the journey matters to them."
        ),
    ),
    Priority(
        label="The experience and memories",
        focus_note=(
            "Focuses on what they saw, felt, and experienced — the wildlife, the views, "
            "a specific moment. The memory is the measure."
        ),
    ),
    Priority(
        label="Reliability and logistics",
        focus_note=(
            "Cares most about things running smoothly — punctuality, pre-arrangement, "
            "no last-minute surprises. Good logistics = good trip for this person. "
            "Write about ONE specific logistics win, not the entire operational checklist."
        ),
    ),
    Priority(
        label="Nature, scenery, and environment",
        focus_note=(
            "Most moved by the forest, the mountains, the river, the birds, the morning mist. "
            "The place itself — not the service — is what they write about most."
        ),
    ),
]


# ── Die 7: Service — correlated with age + travel group ───────────────────────

def _compute_service_weights(age: AgeBucket, group: TravelGroup) -> list[float]:
    """
    Compute service selection weights based on who this person is.
    Order maps to ALL_SERVICES: [safari, hotel, taxi, tour, sightseeing]

    Not random — real people's demographics influence what they book.
    Still probabilistic — a 60yr old CAN book a safari, just less likely.
    """
    # Base weights: [safari, hotel, taxi, tour, sightseeing]
    # Priority order requested: tour, hotel, safari, sightseeing, taxi
    w = [0.20, 0.25, 0.08, 0.35, 0.12]

    # Age adjustments
    if age.label in ("50s", "60s+"):
        w[0] -= 0.08  # deep safari less likely for older travelers
        w[4] += 0.05  # sightseeing more comfortable
        w[3] += 0.03  # tour packages (including pilgrimage) preferred
    elif age.label == "20s":
        w[0] += 0.07  # safari is the exciting option for young travelers
        w[2] += 0.04  # budget cab/taxi very common for young

    # Travel group adjustments
    if group.label == "Family with young children":
        w[3] += 0.10  # pre-arranged tour package most practical
        w[1] += 0.03  # hotel stay important for families
        w[0] -= 0.08  # deep jungle safari less practical with small kids
    elif group.label == "Friends group trip":
        w[0] += 0.08  # safari is the group adventure choice
        w[3] += 0.03
    elif group.label == "Traveling with parents or senior family members":
        w[4] += 0.10  # sightseeing easiest for seniors
        w[3] += 0.04  # pre-arranged tour best for senior travel
        w[0] -= 0.10  # rough safari terrain not ideal for seniors
    elif group.label == "Couple (anniversary or leisure trip)":
        w[1] += 0.04  # hotel/resort stay matters for couples
        w[4] += 0.04  # scenic sightseeing
    elif group.label == "Solo traveler":
        w[2] += 0.08  # taxi/cab most useful for solo travelers
        w[0] += 0.04

    # Clamp minimum probability and normalize to 1.0
    w = [max(0.04, x) for x in w]
    total = sum(w)
    return [x / total for x in w]


# ── PersonaCard ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PersonaCard:
    """
    A fully composed persona for one review generation.
    Contains all 7 dice rolls + optional friction seed + emotional outcome.
    """
    seed: int
    age: AgeBucket             # Die 1
    region: Region             # Die 2
    travel_group: TravelGroup  # Die 3
    personality: Personality   # Die 4
    typing_style: TypingStyle  # Die 5
    priority: Priority         # Die 6
    service: ServiceContext    # Die 7 — correlated, not independent
    length: LengthProfile
    temperature_offset: float
    resolved_context_str: str         # The specific scenario string for this roll
    friction_seed: Optional[FrictionSeed]  # None for 75% of reviews
    emotional_outcome: EmotionalOutcome   # Always rolled — controls emotional landing


# ── Roll function ──────────────────────────────────────────────────────────────

def roll_persona() -> PersonaCard:
    """
    Roll all 7 dice + friction seed + emotional outcome to compose a unique,
    internally coherent persona.

    Dice 1–6 are independent. Die 7 (service) uses weights computed
    from Die 1 (age) + Die 3 (travel group) for realistic correlation.

    Post-roll override: the "anxiety arc" combos (senior/family group +
    reliability priority) are detected and priority is re-rolled to
    prevent the guaranteed problem→solution narrative structure.

    Each call gets a fresh seed — fully reproducible if needed.
    """
    seed = random.randint(10000, 99999)
    rng = random.Random(seed)

    age          = rng.choices(_AGE_BUCKETS,      weights=_AGE_WEIGHTS,          k=1)[0]
    region       = rng.choices(_REGIONS,           weights=_REGION_WEIGHTS,       k=1)[0]
    travel_group = rng.choices(_TRAVEL_GROUPS,     weights=_TRAVEL_GROUP_WEIGHTS, k=1)[0]
    personality  = rng.choice(_PERSONALITIES)
    typing_style = rng.choice(_TYPING_STYLES)
    priority     = rng.choice(_PRIORITIES)
    length       = rng.choices(_LENGTH_PROFILES,   weights=_LENGTH_WEIGHTS,       k=1)[0]

    # ── Fix 5: Break the "anxiety arc" persona combo ───────────────────────────
    # These two groups + "Reliability and logistics" priority = guaranteed
    # problem→solution narrative arc in the LLM output.
    # Re-roll priority to any non-logistics option to prevent this.
    _anxiety_arc_groups = {
        "Traveling with parents or senior family members",
        "Family with young children",
    }
    if (travel_group.label in _anxiety_arc_groups
            and priority.label == "Reliability and logistics"):
        non_logistics = [p for p in _PRIORITIES if p.label != "Reliability and logistics"]
        priority = rng.choice(non_logistics)

    # Die 7: service correlated with age + travel group
    service_weights = _compute_service_weights(age, travel_group)
    service = rng.choices(ALL_SERVICES, weights=service_weights, k=1)[0]

    # Resolve the final specific scenario in Python using the same rng seed
    resolved_context = service.get_context(rng)

    temperature_offset = rng.uniform(-0.10, 0.10)

    # ── Fix 2: Friction Seed (25% probability) ─────────────────────────────────
    friction_seed: Optional[FrictionSeed] = None
    if rng.random() < _FRICTION_PROBABILITY:
        friction_seed = rng.choice(_FRICTION_SEEDS)

    # ── Fix 4: Emotional Outcome (always rolled) ───────────────────────────────
    emotional_outcome = rng.choice(_EMOTIONAL_OUTCOMES)

    return PersonaCard(
        seed=seed,
        age=age,
        region=region,
        travel_group=travel_group,
        personality=personality,
        typing_style=typing_style,
        priority=priority,
        service=service,
        length=length,
        temperature_offset=temperature_offset,
        resolved_context_str=resolved_context,
        friction_seed=friction_seed,
        emotional_outcome=emotional_outcome,
    )


# ── Prompt formatter ───────────────────────────────────────────────────────────

def format_persona_for_prompt(card: PersonaCard) -> str:
    """
    Convert a PersonaCard into a purely descriptive character profile
    for injection into the LLM system prompt.

    Design rule: Describe WHO this person IS and WHERE they should emotionally land.
    Do NOT prescribe specific words, phrases, or Hinglish examples.
    The LLM infers natural language from the character — not from a vocabulary menu.
    This prevents any single phrase from repeating across reviews.
    """
    # Build optional friction block
    friction_block = ""
    if card.friction_seed is not None:
        friction_block = f"""
── MINOR REAL-WORLD DETAIL ──
(Weave this into the review naturally — don't dramatize it, don't apologize for it, don't let it become the main point)
{card.friction_seed.description}
"""

    return f"""── THIS REVIEWER (seed #{card.seed}) ──

WHO THEY ARE:
A person in their {card.age.label}, from {card.region.label}.
Traveling as: {card.travel_group.label}

THEIR CHARACTER:
{card.personality.description}

WHAT MATTERED MOST TO THEM ON THIS TRIP:
{card.priority.focus_note}

HOW THEY WRITE:
{card.typing_style.description}
Cultural language texture: {card.region.language_note}
Age and energy context: {card.age.character_note}
Travel group lens: {card.travel_group.perspective_note}

REVIEW LENGTH: {card.length.label} — {card.length.word_hint}
  → {card.length.description}
  → Aim for {card.length.sentence_range[0]} to {card.length.sentence_range[1]} sentences
{friction_block}
── EMOTIONAL LANDING ──
(This is where the review ends up — the feeling of the last observation)
{card.emotional_outcome.landing_note}

── WRITE AS THIS PERSON ──
Fully inhabit their voice. Adopt their energy level, grammar habits,
sentence rhythm, and cultural background completely.
If their natural style includes small imperfections, let them be.
If they would write three words and stop, stop there.
Do NOT sand down their voice into smooth, polished AI prose.
Do NOT apply a narrative arc (worry → resolution → thanks).
Do NOT wrap up with a conclusion sentence.
Start in the experience. End when the last observation is complete."""
