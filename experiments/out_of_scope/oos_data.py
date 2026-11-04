"""Authored synthetic out-of-scope data, corpus loading and encoding.

Everything in ``FAMILIES``, ``AMBIGUOUS`` and ``MIXED`` was written for this
study. It is synthetic: authors are the study authors, not farmers, and none of
it is a representative sample of real KCC traffic. Every frame carries
``synthetic=True`` and result files repeat that label.

Grouping for leakage control
----------------------------
A *family* is one underlying question with an English, Hindi and Hinglish
realisation. Families are the grouping unit: all three registers of a family
always land in the same split, so a detector can never be tuned on the English
form of a question it is tested on in Hindi.

Corpus side
-----------
In-scope queries come from Batch A's grouped split (no question group spans
splits). Only the **English** register exists in the corpus, which is why no
Hindi/Hinglish in-scope claim can be made.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPED_ART = ROOT / "experiments" / "grouped_evaluation" / "artifacts"
DATA = ROOT / "data" / "processed"
ART = HERE / "artifacts"
RESULTS = ROOT / "results" / "out_of_scope"

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lang import detect_register  # noqa: E402

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42
TRAIN_FAMILIES_PER_CATEGORY = 3
VAL_FAMILIES_PER_CATEGORY = 1
TEST_FAMILIES_PER_CATEGORY = 1

# (category, English, Hindi, Hinglish). One row per family.
FAMILIES = [
    ("programming",
     "Write a Python program to sort a list.",
     "पायथन में सूची को क्रमबद्ध करने का प्रोग्राम लिखें।",
     "Python me list sort karne ka program likho."),
    ("programming",
     "How do I fix an error in my JavaScript code?",
     "मेरे जावास्क्रिप्ट कोड में त्रुटि कैसे ठीक करें?",
     "Mere JavaScript code me error kaise thik karein?"),
    ("programming",
     "Explain how a for loop works in C.",
     "बताइए कि सी भाषा में फॉर लूप कैसे काम करता है।",
     "C language me for loop kaise kaam karta hai samjhao."),
    ("programming",
     "What is the difference between a list and a tuple?",
     "सूची और टपल में क्या अंतर है?",
     "List aur tuple me kya antar hai?"),
    ("programming",
     "Write SQL to find the top ten customers.",
     "शीर्ष दस ग्राहकों को खोजने के लिए एसक्यूएल लिखें।",
     "Top ten customers dhoondhne ki SQL query likho."),

    ("general_knowledge",
     "What is the capital of Japan?",
     "जापान की राजधानी क्या है?",
     "Japan ki rajdhani kya hai?"),
    ("general_knowledge",
     "Who wrote the play Hamlet?",
     "हैमलेट नाटक किसने लिखा था?",
     "Hamlet natak kisne likha tha?"),
    ("general_knowledge",
     "How many continents are there on Earth?",
     "पृथ्वी पर कितने महाद्वीप हैं?",
     "Prithvi par kitne mahadweep hain?"),
    ("general_knowledge",
     "What is the boiling point of water at sea level?",
     "समुद्र तल पर पानी का क्वथनांक क्या है?",
     "Samudra tal par pani ka boiling point kya hai?"),
    ("general_knowledge",
     "Which is the largest ocean in the world?",
     "विश्व का सबसे बड़ा महासागर कौन सा है?",
     "Duniya ka sabse bada ocean kaunsa hai?"),

    ("sports",
     "Who won yesterday's cricket match?",
     "कल का क्रिकेट मैच किसने जीता?",
     "Kal ka cricket match kisne jeeta?"),
    ("sports",
     "What is the offside rule in football?",
     "फुटबॉल में ऑफसाइड नियम क्या है?",
     "Football me offside rule kya hai?"),
    ("sports",
     "When are the next Olympic Games?",
     "अगले ओलंपिक खेल कब होंगे?",
     "Next Olympic games kab honge?"),
    ("sports",
     "How many players are on a basketball team?",
     "बास्केटबॉल टीम में कितने खिलाड़ी होते हैं?",
     "Basketball team me kitne players hote hain?"),
    ("sports",
     "Who is the captain of the Indian hockey team?",
     "भारतीय हॉकी टीम का कप्तान कौन है?",
     "Indian hockey team ka captain kaun hai?"),

    ("human_health",
     "What is a normal blood pressure reading?",
     "सामान्य रक्तचाप कितना होता है?",
     "Normal blood pressure kitna hota hai?"),
    ("human_health",
     "How do I treat a headache at home?",
     "घर पर सिरदर्द का इलाज कैसे करें?",
     "Ghar par sar dard ka ilaj kaise karein?"),
    ("human_health",
     "Which vitamins are good for the eyes?",
     "आंखों के लिए कौन से विटामिन अच्छे हैं?",
     "Aankhon ke liye kaun se vitamin acche hain?"),
    ("human_health",
     "How much water should an adult drink daily?",
     "एक वयस्क को रोज कितना पानी पीना चाहिए?",
     "Ek adult ko roz kitna pani peena chahiye?"),
    ("human_health",
     "What causes a sore throat?",
     "गले में खराश क्यों होती है?",
     "Gale me kharash kyun hoti hai?"),

    ("personal_finance",
     "How do I open a savings account?",
     "बचत खाता कैसे खोलें?",
     "Savings account kaise kholein?"),
    ("personal_finance",
     "What is a mutual fund?",
     "म्यूचुअल फंड क्या है?",
     "Mutual fund kya hai?"),
    ("personal_finance",
     "How is income tax calculated?",
     "आयकर की गणना कैसे होती है?",
     "Income tax kaise calculate hota hai?"),
    ("personal_finance",
     "Should I invest in gold or stocks?",
     "मुझे सोने में निवेश करना चाहिए या शेयरों में?",
     "Mujhe gold me invest karna chahiye ya stocks me?"),
    ("personal_finance",
     "What is the interest rate on a fixed deposit?",
     "सावधि जमा पर ब्याज दर क्या है?",
     "Fixed deposit par interest rate kya hai?"),

    ("entertainment",
     "Which movie won the best picture award?",
     "सर्वश्रेष्ठ फिल्म का पुरस्कार किसे मिला?",
     "Best picture award kis film ko mila?"),
    ("entertainment",
     "What are the best songs of this year?",
     "इस साल के सबसे अच्छे गाने कौन से हैं?",
     "Is saal ke best songs kaun se hain?"),
    ("entertainment",
     "Who is your favourite actor?",
     "आपके पसंदीदा अभिनेता कौन हैं?",
     "Aapke favourite actor kaun hain?"),
    ("entertainment",
     "When does the new web series release?",
     "नई वेब सीरीज कब रिलीज होगी?",
     "Nayi web series kab release hogi?"),
    ("entertainment",
     "Recommend a good comedy show.",
     "कोई अच्छा कॉमेडी शो सुझाएं।",
     "Koi accha comedy show suggest karo."),

    ("politics",
     "Who is the current prime minister?",
     "वर्तमान प्रधानमंत्री कौन हैं?",
     "Current prime minister kaun hain?"),
    ("politics",
     "When is the next election?",
     "अगला चुनाव कब है?",
     "Agla election kab hai?"),
    ("politics",
     "What is the role of parliament?",
     "संसद की भूमिका क्या है?",
     "Parliament ki bhumika kya hai?"),
    ("politics",
     "Which party won the last state election?",
     "पिछले राज्य चुनाव में कौन सी पार्टी जीती?",
     "Pichle state election me kaun si party jeeti?"),
    ("politics",
     "How does a bill become a law?",
     "विधेयक कानून कैसे बनता है?",
     "Bill kanun kaise banta hai?"),

    ("travel",
     "What is the best time to visit Goa?",
     "गोवा घूमने का सबसे अच्छा समय क्या है?",
     "Goa ghoomne ka best time kya hai?"),
    ("travel",
     "How much does a train ticket to Delhi cost?",
     "दिल्ली की ट्रेन टिकट कितने की है?",
     "Delhi ki train ticket kitne ki hai?"),
    ("travel",
     "Which hotel is near the airport?",
     "हवाई अड्डे के पास कौन सा होटल है?",
     "Airport ke paas kaun sa hotel hai?"),
    ("travel",
     "Do I need a visa for Thailand?",
     "थाईलैंड के लिए वीजा चाहिए?",
     "Thailand ke liye visa chahiye?"),
    ("travel",
     "What are good places to visit in Rajasthan?",
     "राजस्थान में घूमने की अच्छी जगहें कौन सी हैं?",
     "Rajasthan me ghoomne ki acchi jagah kaun si hain?"),

    ("education",
     "How can I prepare for the civil services exam?",
     "सिविल सेवा परीक्षा की तैयारी कैसे करें?",
     "Civil services exam ki taiyari kaise karein?"),
    ("education",
     "What is the syllabus for class ten mathematics?",
     "दसवीं गणित का पाठ्यक्रम क्या है?",
     "Class ten maths ka syllabus kya hai?"),
    ("education",
     "Which college is best for engineering?",
     "इंजीनियरिंग के लिए कौन सा कॉलेज सबसे अच्छा है?",
     "Engineering ke liye kaun sa college best hai?"),
    ("education",
     "How do I apply for a scholarship?",
     "छात्रवृत्ति के लिए आवेदन कैसे करें?",
     "Scholarship ke liye apply kaise karein?"),
    ("education",
     "Explain photosynthesis for a school project.",
     "स्कूल प्रोजेक्ट के लिए प्रकाश संश्लेषण समझाइए।",
     "School project ke liye photosynthesis samjhao."),

    ("technology",
     "How does blockchain work?",
     "ब्लॉकचेन कैसे काम करता है?",
     "Blockchain kaise kaam karta hai?"),
    ("technology",
     "Which laptop is best under fifty thousand?",
     "पचास हजार में कौन सा लैपटॉप अच्छा है?",
     "Pachas hazaar me kaun sa laptop accha hai?"),
    ("technology",
     "Why is my phone battery draining fast?",
     "मेरे फोन की बैटरी जल्दी खत्म क्यों होती है?",
     "Mere phone ki battery jaldi khatam kyun hoti hai?"),
    ("technology",
     "How do I create an email account?",
     "ईमेल खाता कैसे बनाएं?",
     "Email account kaise banayein?"),
    ("technology",
     "What is artificial intelligence?",
     "कृत्रिम बुद्धिमत्ता क्या है?",
     "Artificial intelligence kya hai?"),

    ("law",
     "How do I file a police complaint?",
     "पुलिस में शिकायत कैसे दर्ज करें?",
     "Police me complaint kaise darj karein?"),
    ("law",
     "What is the punishment for theft?",
     "चोरी की सजा क्या है?",
     "Chori ki saza kya hai?"),
    ("law",
     "How can I get a divorce?",
     "तलाक कैसे लें?",
     "Talak kaise lein?"),
    ("law",
     "What are my rights as a tenant?",
     "किरायेदार के रूप में मेरे अधिकार क्या हैं?",
     "Kirayedar ke roop me mere adhikar kya hain?"),
    ("law",
     "How do I write a will?",
     "वसीयत कैसे लिखें?",
     "Vasiyat kaise likhein?"),

    ("relationships",
     "How do I apologise to a friend?",
     "दोस्त से माफी कैसे मांगें?",
     "Dost se maafi kaise maangein?"),
    ("relationships",
     "What should I gift my wife on her birthday?",
     "पत्नी को जन्मदिन पर क्या तोहफा दें?",
     "Patni ko birthday par kya gift dein?"),
    ("relationships",
     "How can I improve my relationship with my parents?",
     "माता-पिता के साथ संबंध कैसे सुधारें?",
     "Maa-baap ke saath relation kaise improve karein?"),
    ("relationships",
     "My friend is not talking to me, what should I do?",
     "मेरा दोस्त मुझसे बात नहीं कर रहा, क्या करूं?",
     "Mera dost mujhse baat nahi kar raha, kya karun?"),
    ("relationships",
     "How do I say no without hurting someone?",
     "किसी को दुखाए बिना मना कैसे करें?",
     "Kisi ko dukhaye bina mana kaise karein?"),
]

REGISTERS = ("en", "hi", "hinglish")

# Borderline queries: neither clearly in scope nor clearly unrelated. Never used
# for threshold selection or for headline metrics; reported descriptively.
AMBIGUOUS = [
    "Is it safe to use pesticide spray in my kitchen garden?",
    "My dog has a skin infection, which medicine should I give?",
    "What is the price of tomatoes in the supermarket today?",
    "Can I get a loan for a small shop in my village?",
    "Which smartphone app gives weather alerts?",
    "How do I compost kitchen waste at home?",
    "Is cow milk good for a baby?",
    "How much does a tractor cost?",
    "What is the best fertiliser for a balcony plant?",
    "How do I keep rats out of my house?",
    "Should I buy crop insurance or a health policy?",
    "Which fish is good for eating?",
    "How do I grow grass on my lawn?",
    "What is the mandi rate for onions today?",
    "Is honey good for health?",
    "How do I protect my garden from monkeys?",
]

# Mixed-intent queries: agricultural content plus an unrelated request.
MIXED_INTENT = [
    "My tomato leaves are turning yellow and also write a Python program for me.",
    "Which fertiliser is best for paddy and what is the capital of Japan?",
    "My buffalo is not eating and please book a train ticket to Delhi.",
    "How do I control aphids in mustard and who won yesterday's cricket match?",
    "What is the urea dose for wheat and explain blockchain technology.",
    "My mango tree has white spots and what is a mutual fund?",
    "Tell me the market price of cotton and write a poem about rain.",
    "How do I sow bajra and how do I open a savings account?",
    "My cow has fever and which movie won the award?",
    "What pesticide for bollworm and how do I apply for a passport?",
    "Best seed variety for gram and what is the interest rate on fixed deposit?",
    "My brinjal crop has wilt and give me a recipe for brinjal curry.",
    "How do I irrigate sugarcane and who is the prime minister?",
    "Which fungicide for blight and how do I fix my laptop battery?",
    "My goat is limping and what is the schedule of the next train?",
    "Fertilizer for tomato plants and what is the boiling point of water?",
]


def build_synthetic_frame() -> pd.DataFrame:
    """Long frame: one row per family/register, with the family as group."""
    records = []
    for idx, (category, en, hi, hinglish) in enumerate(FAMILIES, start=1):
        family = f"{category}-{idx:02d}"
        for register, text in zip(REGISTERS, (en, hi, hinglish)):
            records.append({
                "query_id": f"{family}-{register}",
                "family_id": family,
                "category": category,
                "register": register,
                "text": text,
                "label": "out_of_scope",
                "synthetic": True,
                "source": "authored_synthetic",
            })
    frame = pd.DataFrame.from_records(records)
    if frame["text"].duplicated().any():
        dup = frame.loc[frame["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"duplicate synthetic query text: {dup!r}")
    return frame


def build_ambiguous_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": [f"amb-{i:02d}" for i in range(1, len(AMBIGUOUS) + 1)],
        "family_id": [f"amb-{i:02d}" for i in range(1, len(AMBIGUOUS) + 1)],
        "category": "ambiguous",
        "register": [detect_register(t) for t in AMBIGUOUS],
        "text": AMBIGUOUS,
        "label": "ambiguous",
        "synthetic": True,
        "source": "authored_synthetic",
    })


def build_mixed_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": [f"mix-{i:02d}" for i in range(1, len(MIXED_INTENT) + 1)],
        "family_id": [f"mix-{i:02d}" for i in range(1, len(MIXED_INTENT) + 1)],
        "category": "mixed_intent",
        "register": [detect_register(t) for t in MIXED_INTENT],
        "text": MIXED_INTENT,
        "label": "mixed_intent",
        "synthetic": True,
        "source": "authored_synthetic",
    })


def split_synthetic(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign each family to train/val/test, stratified by category.

    The family (all three registers) is the unit, so no paraphrase or
    translation of a test question can appear in training.
    """
    rng = np.random.default_rng(SEED)
    assignment = {}
    for category, block in frame.groupby("category", sort=True):
        families = sorted(block["family_id"].unique())
        families = [families[i] for i in rng.permutation(len(families))]
        if len(families) != (TRAIN_FAMILIES_PER_CATEGORY + VAL_FAMILIES_PER_CATEGORY
                             + TEST_FAMILIES_PER_CATEGORY):
            raise ValueError(f"{category}: expected 5 families, found {len(families)}")
        for family in families[:TRAIN_FAMILIES_PER_CATEGORY]:
            assignment[family] = "train"
        for family in families[TRAIN_FAMILIES_PER_CATEGORY:
                               TRAIN_FAMILIES_PER_CATEGORY + VAL_FAMILIES_PER_CATEGORY]:
            assignment[family] = "val"
        for family in families[-TEST_FAMILIES_PER_CATEGORY:]:
            assignment[family] = "test"
    out = frame.copy()
    out["split"] = out["family_id"].map(assignment)
    if out["split"].isna().any():
        raise ValueError("some synthetic families were not assigned a split")
    spans = out.groupby("family_id")["split"].nunique()
    if (spans > 1).any():
        raise ValueError("a synthetic family spans more than one split")
    return out


def load_corpus_splits() -> tuple:
    """Original corpus with Batch A group ids and grouped split."""
    df = pd.read_csv(DATA / "full_labeled.csv")
    split = pd.read_csv(GROUPED_ART / "corpus_split.csv")
    groups = pd.read_csv(GROUPED_ART / "groups.csv")
    if not (len(df) == len(split) == len(groups)):
        raise SystemExit("corpus, split and group files disagree in length")
    if not np.array_equal(df["qid"].to_numpy(), split["qid"].to_numpy()):
        raise SystemExit("corpus_split.csv does not align with the corpus")
    df = df.copy()
    df["group_id"] = groups["group_id"].to_numpy()
    df["split"] = split["split"].to_numpy()
    spans = df.groupby("group_id")["split"].nunique()
    if (spans > 1).any():
        raise SystemExit("question groups span splits; refusing to evaluate")
    return df


def encode_texts(texts, name: str, force: bool = False) -> np.ndarray:
    """Encode with the project's existing MiniLM model, content-addressed cache."""
    texts = [str(t) for t in texts]
    digest = hashlib.sha1("\x00".join(texts).encode("utf-8")).hexdigest()
    vec_path = ART / f"{name}.npy"
    meta_path = ART / f"{name}.json"
    if vec_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arr = np.load(vec_path)
        if meta.get("sha1") == digest and len(arr) == len(texts):
            return arr.astype("float32")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    arr = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False).astype("float32")
    ART.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, arr)
    meta_path.write_text(json.dumps({
        "name": name, "model": MODEL_NAME, "n": int(len(texts)),
        "dim": int(arr.shape[1]), "sha1": digest, "normalised": True,
    }, indent=2), encoding="utf-8")
    return arr
