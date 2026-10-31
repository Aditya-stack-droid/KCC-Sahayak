"""Domain glossaries used for bootstrapped entity extraction and weak intent labeling.

Crop and pest/disease lists cover the crops most common in Haryana KCC call
volume (wheat, paddy/rice, cotton, mustard, sugarcane, bajra, gram, vegetables,
fruit orchards) plus the generic set seen across the wider all-India corpus so
the glossary still fires on the non-Haryana-labelled rows in the public corpus.
"""

CROPS = [
    "wheat", "paddy", "rice", "cotton", "mustard", "sugarcane", "bajra",
    "gram", "chickpea", "maize", "jowar", "barley", "sunflower", "groundnut",
    "soybean", "arhar", "moong", "urad", "masoor", "lentil", "potato",
    "onion", "tomato", "brinjal", "okra", "cabbage", "cauliflower", "chilli",
    "capsicum", "cucumber", "gourd", "pea", "carrot", "radish", "spinach",
    "garlic", "ginger", "turmeric", "banana", "mango", "guava", "papaya",
    "citrus", "kinnow", "grapes", "pomegranate", "coconut", "apple", "litchi",
    "watermelon", "muskmelon", "fodder", "berseem", "lucern", "sorghum",
]

PESTS_SYMPTOMS = [
    "aphid", "whitefly", "bollworm", "stem borer", "leaf folder", "thrips",
    "mealybug", "mite", "jassid", "hopper", "caterpillar", "armyworm",
    "termite", "nematode", "rust", "blight", "wilt", "rot", "mildew",
    "smut", "blast", "spot", "curl", "mosaic", "yellowing", "wilting",
    "infestation", "disease", "fungus", "virus", "damping off", "canker",
    "scab", "gall", "borer", "weed", "weevil", "locust",
]

QUERY_TYPE_RULES = {
    "Plant Protection": [
        "control measure", "pest", "disease", "infestation", "spray",
        "insecticide", "pesticide", "fungicide", "weedicide", "herbicide",
        "control of", "attack of", "treatment of",
    ] + PESTS_SYMPTOMS,
    "Nutrient Management": [
        "fertilizer", "fertiliser", "dose", "urea", "dap", "ssp", "mop",
        "manure", "fym", "nutrient", "micronutrient", "zinc", "compost",
    ],
    "Weather": [
        "weather", "rainfall", "rain", "temperature", "forecast", "climate",
        "humidity", "frost", "hailstorm",
    ],
    "Government Scheme": [
        "kisan credit card", "credit card", "loan", "subsidy", "insurance",
        "scheme", "pm kisan", "pmfby", "compensation", "yojana", "aadhar",
        "adhar", "bank", "sbi",
    ],
    "Market Price": [
        "price", "market", "mandi", "rate of", "selling", "msp",
        "procurement",
    ],
    "Varietal Information": [
        "variety", "seed", "hybrid", "source of", "sowing time",
        "germination",
    ],
    "Animal Husbandry": [
        "cow", "buffalo", "cattle", "milk", "goat", "poultry", "livestock",
        "veterinary", "fodder", "vaccination", "calf",
    ],
    "Fisheries": [
        "fish", "fishery", "pond", "breeding of fish",
    ],
    "Cultural Practices": [
        "sowing", "irrigation", "pruning", "spacing", "transplanting",
        "harvesting", "intercropping", "training",
    ],
}

# Categories where a wrong automated answer carries real agronomic/financial
# risk (dosage, chemical names, credit/finance decisions). These are always
# escalated regardless of model confidence.
HIGH_STAKES_TYPES = {"Plant Protection", "Nutrient Management", "Government Scheme"}

# Categories whose correct answer changes day to day (forecasts, mandi rates).
# A logged answer from an earlier call is stale by construction, so retrieval can
# never answer these correctly; they need live data or a human.
TIME_SENSITIVE_TYPES = {"Weather", "Market Price"}
