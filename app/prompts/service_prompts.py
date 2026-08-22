"""
Service-specific context blocks for Gola Holidays review generation.

Each ServiceContext provides factual background about ONE service.
To prevent the LLM from deterministically picking the first bullet point every time,
the specific scenario/route/hotel is chosen randomly in Python via `get_context(rng)`.
The LLM only ever sees ONE scenario to write about per generation.

v2 changes (post-audit):
  - Rewrote all base_context blocks from "WHAT GOLA ARRANGED: ..." (company-as-hero)
    to "YOUR EXPERIENCE: ..." (customer-as-subject) framing
  - Removed city attribution ("Ramnagar, Uttarakhand") from every base_context block
    to prevent it from appearing as a planted SEO phrase in every review
  - Explicitly directs the LLM to write from the customer's sensory/experiential
    perspective rather than cataloguing the business's service capabilities
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceContext:
    """Factual context about a single Gola Holidays service."""
    service_id: str       # e.g., "safari", "hotel", "taxi"
    service_name: str     # human-readable name
    base_context: str     # general background info
    scenarios: list[str]  # specific routes, zones, or packages

    def get_context(self, rng: random.Random) -> str:
        """Pick a specific scenario randomly in Python so the LLM doesn't just pick the first one."""
        scenario = rng.choice(self.scenarios)
        return (
            f"{self.base_context}\n\n"
            f"SPECIFIC EXPERIENCE TO WRITE ABOUT (focus only on this one):\n- {scenario}"
        )


# ── SERVICE 1: JIM CORBETT SAFARI ─────────────────────────────────────────────

_SAFARI = ServiceContext(
    service_id="safari",
    service_name="Jim Corbett Safari",
    base_context=(
        "SERVICE TYPE: Jim Corbett National Park jungle safari.\n"
        "HOW IT WORKS: You get a jeep or canter, a forest guide, and a zone-specific permit "
        "for a timed slot inside the park. Booking these through a local operator saves "
        "the offline permit process.\n"
        "WRITE FROM: Your experience inside the jungle — what you saw, heard, felt. "
        "The forest, the animals, the early morning drive. "
        "Not a list of what was 'arranged' — your actual time in the jungle."
    ),
    scenarios=[
        "Bijrani Zone (Jeep Safari) — dense sal forest, high tiger activity, spotted a tiger, jeep picked us up from resort",
        "Jhirna Zone (Jeep Safari) — open grassland, saw elephants and leopard, morning slot",
        "Garjia Zone (Jeep Safari) — near Garjia Devi Temple, lots of birds and spotted deer, resort pickup",
        "Dhikala Zone (Day Visit via Canter Safari) — shared open Canter bus starting from Ramnagar town (no resort pickup), deep core zone, very limited permits, saw a tusker",
        "Durga Devi Zone (Jeep Safari) — remote, river views, less crowded, crocodile on Ramganga riverbank",
        "Sitabani Zone (Jeep Safari) — buffer zone, no permit needed, quieter, peacocks and hornbills",
        "Forest Rest House stay (Night Stay inside Dhikala) — sleeping inside the core park, exclusive jeep safari inside, jungle sounds at night",
    ],
)


# ── SERVICE 2: HOTEL / RESORT STAY ────────────────────────────────────────────

_HOTEL = ServiceContext(
    service_id="hotel",
    service_name="Hotel / Resort Booking",
    base_context=(
        "SERVICE TYPE: Hotel or resort stay.\n"
        "HOW IT WORKS: A local travel operator matches a property to your group, budget, "
        "and location requirement, and pre-confirms the booking so there are no surprises "
        "at check-in.\n"
        "WRITE FROM: Your actual stay — what the property was like, what you noticed from "
        "your room or the grounds, something specific about the location or atmosphere. "
        "Not a checklist of services. One or two things that stuck with you."
    ),
    scenarios=[
        "Corbett: Jungle-facing resort — wildlife sounds at night, forest atmosphere",
        "Corbett: River-facing resort on Kosi or Ramganga — water sounds, outdoor seating",
        "Corbett: Budget lodge in Ramnagar town — clean, simple, practical",
        "Nainital: Lake-view hotel, short walking distance to Mall Road",
        "Bhimtal: Quieter lakeside property, peaceful and away from city noise",
        "Rishikesh: Ganga view property, calm and spiritual atmosphere",
        "Haridwar: Hotel near the ghats, easy access for evening aarti",
        "Auli: Mountain resort with snow views and Nanda Devi peak backdrop",
        "Mussoorie: Hillside property with valley view, near Mall Road",
        "Munsiyari: Remote Himalayan stay, Panchachuli peaks visible on clear mornings",
    ],
)


# ── SERVICE 3: TAXI / CAB SERVICE ─────────────────────────────────────────────

_TAXI = ServiceContext(
    service_id="taxi",
    service_name="Taxi / Cab Service",
    base_context=(
        "SERVICE TYPE: Taxi or cab for a specific route.\n"
        "HOW IT WORKS: A vehicle and driver are pre-assigned for a pickup from a station, "
        "airport, or hotel for a specific route. Local operators know the mountain roads well.\n"
        "WRITE FROM: Your experience of the journey — the drive, the vehicle, the roads, "
        "what you saw along the way. The mountain terrain, the hairpin bends, the views. "
        "Not a logistics summary — the actual feel of being in that car on that road."
    ),
    scenarios=[
        "Airport Transfer: Pantnagar Airport → Ramnagar / Corbett (Swift Dzire or Etios)",
        "Airport Transfer: Jolly Grant Airport (Dehradun) → Rishikesh / Haridwar (Innova)",
        "Railway Transfer: Kathgodam Station → Nainital / Bhimtal (Innova Crysta)",
        "Railway Transfer: Kathgodam Station → Ramnagar / Corbett (Swift Dzire)",
        "Railway Transfer: Ramnagar Station → Corbett resort (Tempo Traveller)",
        "Long Route: Delhi → Nainital overnight drive (Innova Crysta)",
        "Long Route: Delhi → Haridwar / Rishikesh (Swift Dzire)",
        "Long Route: Kathgodam → Munsiyari long scenic mountain drive (Innova Crysta)",
        "Long Route: Ramnagar → Corbett safari zones (Local Jeep)",
    ],
)


# ── SERVICE 4: TOUR PACKAGES ───────────────────────────────────────────────────

_TOUR = ServiceContext(
    service_id="tour",
    service_name="Tour Package (Multi-day)",
    base_context=(
        "SERVICE TYPE: Multi-day tour package.\n"
        "HOW IT WORKS: A pre-built itinerary covering multiple days, with accommodation, "
        "transport, and activities coordinated in advance. Permits and guides are included "
        "where needed.\n"
        "WRITE FROM: A specific moment, place, or day from the trip — not the overall "
        "package logistics. Pick the one thing that stays with you: a shrine at dawn, "
        "a view from a meadow, a meal on the road, a moment with the group. "
        "Let the trip breathe — don't summarize every stop."
    ),
    scenarios=[
        "Wildlife: Jim Corbett 2N/3D — stay + safari + local sightseeing",
        "Hill Station: Nainital 3N/4D — Naini Lake, Tiffin Top, Mall Road, Bhimtal day trip",
        "Hill Station: Bhimtal + Naukuchiatal 2N/3D — quieter lakes circuit",
        "Hill Station: Mussoorie 2N/3D — Kempty Falls, Company Garden, Cable Car",
        "Hill Station: Munsiyari 4N/5D — 'Little Kashmir', Khaliya Top, Milam Glacier base",
        "Hill Station: Auli 3N/4D — skiing in winter, meadows in summer, Nanda Devi views",
        "Hill Station: Ranikhet 2N/3D — peaceful cantonment, apple orchards",
        "Hill Station: Kausani 2N/3D — sunrise over Himalayan peaks",
        "Pilgrimage: Kedarnath Yatra — helicopter or trek option",
        "Pilgrimage: Char Dham Yatra — Yamunotri, Gangotri, Kedarnath, Badrinath",
        "Pilgrimage: Haridwar + Rishikesh 2N/3D spiritual trip",
        "Adventure: Valley of Flowers + Hemkund Sahib trekking package",
    ],
)


# ── SERVICE 5: LOCAL SIGHTSEEING ──────────────────────────────────────────────

_SIGHTSEEING = ServiceContext(
    service_id="sightseeing",
    service_name="Local Sightseeing",
    base_context=(
        "SERVICE TYPE: Local sightseeing trip with a cab and driver for the day.\n"
        "HOW IT WORKS: A vehicle covers the key local spots in the area at a comfortable "
        "pace. The driver knows the local routes and timings.\n"
        "WRITE FROM: The actual place — what it looked like, how it felt to be there, "
        "what you or your group noticed. A specific spot, a specific moment. "
        "Not 'we visited X Y Z' — what ONE of those places was actually like."
    ),
    scenarios=[
        "Corbett Area: Garjia Devi Temple — hillside temple on a rock in the Kosi river",
        "Corbett Area: Corbett Falls — waterfall in the forest, short trail",
        "Corbett Area: Corbett Museum (Choti Haldwani) — Jim Corbett's old home",
        "Corbett Area: Ramganga river viewpoint — popular sunset spot",
        "Nainital: Naini Lake boat ride and Mall Road stroll",
        "Nainital: Tiffin Top (Dorothy's Seat) — panoramic Himalayan view",
        "Nainital: Snow View Point (cable car) and Naina Devi Temple",
        "Nainital: Day trip covering Bhimtal Lake, Sattal, and Naukuchiatal",
    ],
)


# ── All services list ──────────────────────────────────────────────────────────
# Used by review_randomizer._compute_service_weights()
# Order matters: [safari, hotel, taxi, tour, sightseeing]

ALL_SERVICES: list[ServiceContext] = [_SAFARI, _HOTEL, _TAXI, _TOUR, _SIGHTSEEING]
