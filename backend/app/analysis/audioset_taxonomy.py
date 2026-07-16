from __future__ import annotations

# Primary-speaker speech and non-noise acoustic classes: excluded from noise ranking.
SPEECH_LABELS: frozenset[str] = frozenset({
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Speech synthesizer",
    "Human voice",
})

# Classes that are neither speech nor meaningful background noise.
NON_NOISE_LABELS: frozenset[str] = frozenset({
    "Silence",
    "Inside, small room",
    "Inside, large room or hall",
    "Inside, public space",
})

# Telephony / call-line signalling: INTRINSIC to phone-call audio, not background noise.
# Empirically these dominate the AST output on real calls (Sidetone/Dial tone ~0.7 on our
# samples) and would otherwise flag every call as "noise". Sidetone is the caller's own
# voice fed back into the earpiece; dial/busy/ringtone are line signals; "Telephone" is the
# medium itself. Excluded exactly like speech so the real background noise surfaces.
TELEPHONY_LABELS: frozenset[str] = frozenset({
    "Telephone",
    "Telephone bell ringing",
    "Ringtone",
    "Telephone dialing, DTMF",
    "Dial tone",
    "Busy signal",
    "Sidetone",
    "Cellphone buzz, vibrating alert",
})

EXCLUDED_LABELS: frozenset[str] = SPEECH_LABELS | NON_NOISE_LABELS | TELEPHONY_LABELS

# Curated AudioSet label -> concise noise_type string (spec-aligned vocabulary).
NOISE_ALIASES: dict[str, str] = {
    # Media
    "Television": "TV",
    "Radio": "radio",
    "Music": "music",
    # Chatter / crowd
    "Hubbub, speech noise, speech babble": "office chatter",
    "Chatter": "office chatter",
    "Crowd": "background chatter",
    # Typing / office
    "Typing": "keyboard typing",
    "Computer keyboard": "keyboard typing",
    "Typewriter": "keyboard typing",
    # Static / broadband noise
    "White noise": "static",
    "Pink noise": "static",
    "Static": "static",
    "Noise": "noise",
    "Environmental noise": "noise",
    "Cacophony": "noise",
    # Wind
    "Wind": "wind",
    "Wind noise (microphone)": "wind",
    "Rustling leaves": "wind",
    # Road / vehicles
    "Vehicle": "road noise",
    "Car": "road noise",
    "Traffic noise, roadway noise": "road noise",
    "Motor vehicle (road)": "road noise",
    # Mechanical
    "Engine": "mechanical noise",
    "Mechanisms": "mechanical noise",
    "Machine": "mechanical noise",
    "Air conditioning": "mechanical noise",
    "Hum": "electrical hum",
    "Mains hum": "electrical hum",
}


def is_excluded(label: str) -> bool:
    """True if a label is speech or a non-noise class (excluded from noise ranking)."""
    return label in EXCLUDED_LABELS


def noise_type_for(label: str) -> str:
    """Map an AudioSet label to a concise noise_type string (fallback: lowercased label)."""
    return NOISE_ALIASES.get(label, label.lower())
