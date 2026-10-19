"""Allowed tag values and synonyms. A VLM value is kept only if it is, or maps to, an allowed value.

Changing any list or synonym changes what gets saved, so bump VOCAB_VERSION (part of TAGGER_VERSION)
and re-run scripts/tag_catalogue.py.
"""

VOCAB_VERSION = "1"

# Same names as data/categories.txt
CATEGORY = frozenset({
    "rings", "earrings", "necklaces", "pendants", "bangles", "bracelets", "chains", "nose_pins", "anklets",
})

METAL = frozenset({
    "gold", "white gold", "rose gold", "silver", "platinum", "black metal", "brass", "bronze", "copper", "steel",
})

GEMSTONES = frozenset({
    "diamond", "ruby", "emerald", "sapphire", "pearl", "amethyst", "topaz", "garnet", "opal", "turquoise", "coral",
    "amber", "agate", "onyx", "quartz", "crystal", "cubic zirconia", "peridot", "citrine", "aquamarine",
    "tanzanite", "jade", "lapis lazuli", "tiger's eye", "moonstone", "glass", "shell",
})

STONE_COLORS = frozenset({
    "white", "clear", "black", "red", "pink", "blue", "green", "yellow", "orange", "purple", "brown", "grey",
    "multicolour",
})

STYLE = frozenset({
    "classic", "modern", "traditional", "ethnic", "minimalist", "statement", "vintage", "boho", "delicate",
    "religious", "novelty", "casual", "bridal", "mens",
})

# PROVISIONAL: motifs are open-ended; this list comes from the 65 sample products and Part A replies and is
# expected to change once real client data is tagged. Do not build hard filters on motifs yet.
MOTIFS = frozenset({
    "flower", "leaf", "heart", "star", "crescent", "cross", "evil eye", "animal", "bird", "butterfly", "snake",
    "knot", "infinity", "geometric", "circle", "square", "triangle", "teardrop", "wave", "swirl", "scroll",
    "filigree", "lattice", "cluster", "halo", "solitaire", "bar", "beads", "chain", "coin", "disc", "tassel",
    "feather", "peacock", "paisley", "cutout", "twist", "pave", "cactus", "shell", "chandelier", "kite", "loop",
    "layered", "cabochon", "inlay", "hammered", "mesh", "dome",
})
PROVISIONAL_FIELDS = frozenset({"motifs"})

LIST_FIELDS = ("metal", "gemstones", "stone_colors", "motifs")
ALLOWED = {
    "category": CATEGORY, "metal": METAL, "gemstones": GEMSTONES, "stone_colors": STONE_COLORS,
    "motifs": MOTIFS, "style": STYLE,
}

SYNONYMS = {
    "category": {
        "ring": "rings", "earring": "earrings", "necklace": "necklaces", "pendant": "pendants",
        "bangle": "bangles", "bracelet": "bracelets", "chain": "chains", "nose pin": "nose_pins",
        "nose ring": "nose_pins", "nose stud": "nose_pins", "anklet": "anklets", "studs": "earrings",
        "stud earrings": "earrings", "cuff": "bracelets",
    },
    "metal": {
        "yellow gold": "gold", "gold-tone": "gold", "gold tone": "gold", "gold plated": "gold", "gold-plated": "gold",
        "14k gold": "gold", "18k gold": "gold", "22k gold": "gold", "24k gold": "gold", "golden": "gold",
        "rose-gold": "rose gold", "pink gold": "rose gold", "rose gold-tone": "rose gold",
        "sterling silver": "silver", "silver-tone": "silver", "silver tone": "silver", "oxidized silver": "silver",
        "oxidised silver": "silver", "rhodium": "white gold",
        "black rhodium": "black metal", "blackened metal": "black metal", "oxidized metal": "black metal",
        "gunmetal": "black metal", "black-tone": "black metal",
        "bronze-tone": "bronze", "antique bronze": "bronze", "antique gold": "gold",
        "stainless steel": "steel", "titanium": "steel",
    },
    "gemstones": {
        "diamonds": "diamond", "cz": "cubic zirconia", "cubic zirconium": "cubic zirconia",
        "pink sapphire": "sapphire", "blue sapphire": "sapphire", "smoky quartz": "quartz", "rose quartz": "quartz",
        "black onyx": "onyx", "green onyx": "onyx", "rhinestone": "crystal", "swarovski crystal": "crystal",
        "dichroic glass": "glass", "art glass": "glass", "murano glass": "glass", "blue glass": "glass",
        "mother of pearl": "shell", "mother-of-pearl": "shell", "freshwater pearl": "pearl",
        "tigers eye": "tiger's eye", "tiger eye": "tiger's eye", "lapis": "lapis lazuli",
    },
    "stone_colors": {
        "colorless": "clear", "colourless": "clear", "transparent": "clear", "gray": "grey", "silver": "grey",
        "violet": "purple", "lavender": "purple", "lilac": "purple", "golden": "yellow", "gold": "yellow",
        "champagne": "yellow", "beige": "white", "cream": "white", "ivory": "white", "navy": "blue", "teal": "blue",
        "turquoise": "blue", "maroon": "red", "crimson": "red", "burgundy": "red", "multicolor": "multicolour",
        "multicolored": "multicolour", "multicoloured": "multicolour", "multi-colored": "multicolour",
        "multi-coloured": "multicolour", "rainbow": "multicolour",
    },
    "style": {
        "contemporary": "modern", "geometric": "modern", "elegant": "classic", "timeless": "classic",
        "bohemian": "boho", "antique": "vintage", "retro": "vintage", "art deco": "vintage", "temple": "traditional",
        "simple": "minimalist", "minimal": "minimalist", "dainty": "delicate", "bold": "statement",
        "chunky": "statement", "indian": "ethnic", "tribal": "ethnic", "wedding": "bridal", "engagement": "bridal",
        "men's": "mens", "masculine": "mens", "playful": "novelty", "quirky": "novelty", "whimsical": "novelty",
        "everyday": "casual",
    },
    "motifs": {
        "floral": "flower", "petals": "flower", "clover": "flower", "quatrefoil": "flower", "leaves": "leaf",
        "spiral": "swirl", "scrollwork": "scroll", "pear": "teardrop", "drop": "teardrop", "drops": "teardrop",
        "cut-out": "cutout", "openwork": "cutout", "hole": "cutout", "bead": "beads", "beaded": "beads",
        "bead chain": "beads", "discs": "disc", "circles": "circle", "concentric circles": "circle", "round": "circle",
        "infinity knot": "infinity", "celtic knot": "knot", "zigzag": "geometric", "parallelogram": "geometric",
        "rectangle": "geometric", "hexagon": "geometric", "checkerboard": "lattice", "floral filigree": "filigree",
        "twisted": "twist", "braided": "twist", "herringbone": "chain", "double link": "chain", "links": "chain",
        "leopard": "animal", "elephant": "animal", "loops": "loop", "evil-eye": "evil eye", "moon": "crescent",
    },
}

# Light/dark shades map to the base colour: "light blue" -> "blue"
_COLOUR_MODIFIERS = ("light ", "dark ", "deep ", "pale ", "bright ", "royal ", "sky ", "baby ", "hot ", "soft ")


def normalize(field: str, value) -> str | None:
    """The allowed value for a raw VLM value, or None if it is not in the vocabulary."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().lower().replace("_", " ").split())
    if field == "stone_colors":
        for modifier in _COLOUR_MODIFIERS:
            if text.startswith(modifier):
                text = text[len(modifier):]
    allowed, synonyms = ALLOWED[field], SYNONYMS.get(field, {})
    for candidate in (text, text.replace(" ", "_")):
        if candidate in allowed:
            return candidate
        if candidate in synonyms:
            return synonyms[candidate]
    # simple plurals: "hearts" -> "heart", "crosses" -> "cross", "rubies" -> "ruby"
    for suffix, replacement in (("ies", "y"), ("es", ""), ("s", "")):
        if text.endswith(suffix) and text[: -len(suffix)] + replacement in allowed:
            return text[: -len(suffix)] + replacement
    return None
