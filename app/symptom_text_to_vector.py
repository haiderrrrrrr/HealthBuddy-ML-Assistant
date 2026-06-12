import numpy as np
import difflib
import re
import unicodedata

# Comprehensive SYMPTOM_MAP based on Symcat dataset (376 symptoms for 400+ diseases)
# Symptoms are organized by body system for clarity
SYMPTOM_MAP = {
    # General/Constitutional Symptoms (100-199)
    "fever": 101,
    "chills": 108,
    "fatigue": 110,
    "weakness": 111,
    "malaise": 112,
    "night sweats": 113,
    "sweating": 114,
    "loss of appetite": 115,
    "weight loss": 116,
    "weight gain": 117,
    "dehydration": 118,
    "excessive thirst": 119,
    "pale skin": 120,
    "flushed skin": 121,
    "hot flashes": 122,
    "cold intolerance": 123,
    "heat intolerance": 124,
    
    # Neurological Symptoms (200-299)
    "headache": 201,
    "dizziness": 202,
    "vertigo": 203,
    "lightheadedness": 204,
    "fainting": 205,
    "syncope": 206,
    "seizures": 207,
    "tremor": 208,
    "numbness": 209,
    "tingling": 210,
    "burning sensation": 211,
    "weakness in limbs": 212,
    "paralysis": 213,
    "loss of coordination": 214,
    "balance problems": 215,
    "confusion": 216,
    "disorientation": 217,
    "memory loss": 218,
    "difficulty concentrating": 219,
    "brain fog": 220,
    "slurred speech": 221,
    "difficulty speaking": 222,
    "vision changes": 223,
    "blurred vision": 224,
    "double vision": 225,
    "loss of vision": 226,
    "sensitivity to light": 227,
    "sensitivity to sound": 228,
    "ringing in ears": 229,
    "tinnitus": 230,
    "hearing loss": 231,
    
    # Mental Health Symptoms (300-349)
    "anxiety": 301,
    "nervousness": 302,
    "panic attacks": 303,
    "depression": 304,
    "sadness": 305,
    "irritability": 306,
    "mood swings": 307,
    "restlessness": 308,
    "insomnia": 309,
    "difficulty sleeping": 310,
    "excessive sleepiness": 311,
    "nightmares": 312,
    "hallucinations": 313,
    "paranoia": 314,
    "agitation": 315,
    "emotional lability": 316,
    
    # Head and Neck Symptoms (350-399)
    "sore throat": 351,
    "throat pain": 352,
    "difficulty swallowing": 353,
    "painful swallowing": 354,
    "hoarseness": 355,
    "voice changes": 356,
    "neck pain": 357,
    "neck stiffness": 358,
    "swollen lymph nodes": 359,
    "swollen glands": 360,
    "jaw pain": 361,
    "ear pain": 362,
    "earache": 363,
    "ear discharge": 364,
    "facial pain": 365,
    "sinus pain": 366,
    "sinus pressure": 367,
    "toothache": 368,
    "mouth sores": 369,
    "dry mouth": 370,
    "bad breath": 371,
    "bleeding gums": 372,
    "tongue pain": 373,
    
    # Respiratory Symptoms (400-449)
    "runny nose": 401,
    "nasal congestion": 402,
    "stuffy nose": 403,
    "sneezing": 404,
    "postnasal drip": 405,
    "nosebleed": 406,
    "cough": 407,
    "dry cough": 408,
    "productive cough": 409,
    "coughing up blood": 410,
    "hemoptysis": 411,
    "wheezing": 412,
    "shortness of breath": 413,
    "difficulty breathing": 414,
    "rapid breathing": 415,
    "shallow breathing": 416,
    "labored breathing": 417,
    "choking sensation": 418,
    "chest tightness": 419,
    "apnea": 420,
    "hiccups": 421,
    
    # Cardiovascular Symptoms (450-499)
    "chest pain": 451,
    "chest pressure": 452,
    "chest discomfort": 453,
    "palpitations": 454,
    "rapid heartbeat": 455,
    "irregular heartbeat": 456,
    "slow heartbeat": 457,
    "racing heart": 458,
    "heart flutter": 459,
    "blue lips": 460,
    "blue fingers": 461,
    "cold hands": 462,
    "cold feet": 463,
    "leg pain with walking": 464,
    "calf pain": 465,
    
    # Gastrointestinal Symptoms (500-599)
    "nausea": 501,
    "vomiting": 502,
    "throwing up": 503,
    "dry heaving": 504,
    "abdominal pain": 505,
    "stomach pain": 506,
    "belly pain": 507,
    "cramping": 508,
    "abdominal cramping": 509,
    "bloating": 510,
    "gas": 511,
    "flatulence": 512,
    "belching": 513,
    "heartburn": 514,
    "acid reflux": 515,
    "indigestion": 516,
    "diarrhea": 517,
    "loose stools": 518,
    "watery stools": 519,
    "constipation": 520,
    "difficulty passing stool": 521,
    "blood in stool": 522,
    "black stools": 523,
    "tarry stools": 524,
    "mucus in stool": 525,
    "rectal bleeding": 526,
    "rectal pain": 527,
    "anal pain": 528,
    "hemorrhoids": 529,
    "loss of bowel control": 530,
    "incontinence": 531,
    "changes in bowel habits": 532,
    "early satiety": 533,
    "feeling full quickly": 534,
    
    # Liver/Gallbladder Symptoms (600-619)
    "jaundice": 601,
    "yellow skin": 602,
    "yellow eyes": 603,
    "dark urine": 604,
    "pale stools": 605,
    "right upper quadrant pain": 606,
    "liver pain": 607,
    
    # Urinary Symptoms (620-669)
    "urinary frequency": 621,
    "frequent urination": 622,
    "urinary urgency": 623,
    "burning urination": 624,
    "painful urination": 625,
    "dysuria": 626,
    "difficulty urinating": 627,
    "weak urine stream": 628,
    "inability to urinate": 629,
    "urinary retention": 630,
    "blood in urine": 631,
    "hematuria": 632,
    "cloudy urine": 633,
    "foul-smelling urine": 634,
    "excessive urination": 635,
    "polyuria": 636,
    "bedwetting": 637,
    "urinary incontinence": 638,
    "leaking urine": 639,
    "flank pain": 640,
    "kidney pain": 641,
    
    # Reproductive Symptoms - Female (670-699)
    "vaginal discharge": 671,
    "abnormal vaginal discharge": 672,
    "vaginal bleeding": 673,
    "irregular periods": 674,
    "missed period": 675,
    "heavy periods": 676,
    "painful periods": 677,
    "menstrual cramps": 678,
    "pelvic pain": 679,
    "vaginal itching": 680,
    "vaginal burning": 681,
    "breast pain": 682,
    "breast tenderness": 683,
    "breast lump": 684,
    "nipple discharge": 685,
    "hot flashes during menopause": 686,
    
    # Reproductive Symptoms - Male (700-719)
    "testicular pain": 701,
    "scrotal swelling": 702,
    "penile discharge": 703,
    "erectile dysfunction": 704,
    "painful ejaculation": 705,
    "blood in semen": 706,
    
    # Musculoskeletal Symptoms (720-799)
    "muscle pain": 721,
    "myalgia": 722,
    "muscle aches": 723,
    "muscle cramps": 724,
    "muscle spasms": 725,
    "muscle weakness": 726,
    "muscle stiffness": 727,
    "joint pain": 728,
    "arthralgia": 729,
    "joint stiffness": 730,
    "joint swelling": 731,
    "joint redness": 732,
    "joint warmth": 733,
    "back pain": 734,
    "lower back pain": 735,
    "upper back pain": 736,
    "neck and back pain": 737,
    "shoulder pain": 738,
    "arm pain": 739,
    "elbow pain": 740,
    "wrist pain": 741,
    "hand pain": 742,
    "finger pain": 743,
    "hip pain": 744,
    "knee pain": 745,
    "leg pain": 746,
    "ankle pain": 747,
    "foot pain": 748,
    "heel pain": 749,
    "toe pain": 750,
    "bone pain": 751,
    "limping": 752,
    "difficulty walking": 753,
    "limited range of motion": 754,
    
    # Skin Symptoms (800-849)
    "rash": 801,
    "skin rash": 802,
    "red rash": 803,
    "itchy rash": 804,
    "itching": 805,
    "pruritus": 806,
    "hives": 807,
    "urticaria": 808,
    "skin lesions": 809,
    "skin bumps": 810,
    "blisters": 811,
    "vesicles": 812,
    "pustules": 813,
    "papules": 814,
    "nodules": 815,
    "skin ulcers": 816,
    "open sores": 817,
    "skin discoloration": 818,
    "bruising": 819,
    "easy bruising": 820,
    "petechiae": 821,
    "purpura": 822,
    "skin peeling": 823,
    "dry skin": 824,
    "scaly skin": 825,
    "cracked skin": 826,
    "skin thickening": 827,
    "mole changes": 828,
    "new moles": 829,
    "warts": 830,
    "skin tags": 831,
    "acne": 832,
    "boils": 833,
    "abscess": 834,
    "cellulitis": 835,
    "swelling": 836,
    "edema": 837,
    "leg swelling": 838,
    "ankle swelling": 839,
    "facial swelling": 840,
    "swollen eyes": 841,
    "hair loss": 842,
    "alopecia": 843,
    "nail changes": 844,
    "brittle nails": 845,
    
    # Eye Symptoms (850-879)
    "red eyes": 851,
    "bloodshot eyes": 852,
    "eye pain": 853,
    "eye irritation": 854,
    "eye itching": 855,
    "eye discharge": 856,
    "watery eyes": 857,
    "dry eyes": 858,
    "crusty eyes": 859,
    "swollen eyelids": 860,
    "drooping eyelid": 861,
    "eye twitching": 862,
    "floaters": 863,
    "flashes of light": 864,
    "eye strain": 865,
    "tired eyes": 866,
    
    # Bleeding/Hematologic Symptoms (880-899)
    "bleeding": 881,
    "easy bleeding": 882,
    "prolonged bleeding": 883,
    "nosebleeds": 884,
    "bleeding gums": 885,
    "blood in vomit": 886,
    "vomiting blood": 887,
    "hematemesis": 888,
    
    # Allergy/Immune Symptoms (900-929)
    "allergic reaction": 901,
    "facial swelling from allergy": 902,
    "tongue swelling": 903,
    "throat swelling": 904,
    "difficulty breathing from allergy": 905,
    "anaphylaxis symptoms": 906,
    "frequent infections": 907,
    "slow healing": 908,
    "swollen lymph nodes generalized": 909,
    
    # Endocrine Symptoms (930-959)
    "increased hunger": 931,
    "excessive hunger": 932,
    "decreased appetite": 933,
    "excessive thirst": 934,
    "increased urination and thirst": 935,
    "cold intolerance": 936,
    "heat intolerance": 937,
    "growth changes": 938,
    "delayed puberty": 939,
    
    # Pregnancy-Related Symptoms (960-979)
    "morning sickness": 961,
    "nausea in pregnancy": 962,
    "breast changes in pregnancy": 963,
    "frequent urination in pregnancy": 964,
    "pelvic pressure": 965,
    "contractions": 966,
    "water breaking": 967,
    "vaginal bleeding in pregnancy": 968,
    
    # Infectious Disease Symptoms (980-999)
    "body aches": 981,
    "flu-like symptoms": 982,
    "swollen glands": 983,
    "enlarged lymph nodes": 984,
    "stiff neck": 985,
    "sensitivity to light with headache": 986,
    "petechial rash": 987,
    
    # Trauma/Injury Symptoms (1000-1019)
    "bruising after injury": 1001,
    "swelling after injury": 1002,
    "deformity": 1003,
    "inability to move limb": 1004,
    "grinding sensation": 1005,
    "popping sound": 1006,
    "instability": 1007,
    "locking joint": 1008,
    
    # Systemic/Other Symptoms (1020-1099)
    "malaise": 1021,
    "feeling unwell": 1022,
    "decreased energy": 1023,
    "loss of stamina": 1024,
    "decreased exercise tolerance": 1025,
    "swollen feet": 1026,
    "swollen hands": 1027,
    "generalized swelling": 1028,
    "foul body odor": 1029,
    "changes in taste": 1030,
    "metallic taste": 1031,
    "loss of taste": 1032,
    "loss of smell": 1033,
    "anosmia": 1034,
    "increased sensitivity to pain": 1035,
    "decreased sensation": 1036,
    "pins and needles": 1037,
    "crawling sensation": 1038,
}

# Extended synonyms mapping
SYNONYMS = {
    # Fever synonyms
    "pyrexia": "fever",
    "febrile": "fever",
    "feverish": "fever",
    "high temperature": "fever",
    "elevated temperature": "fever",
    
    # Headache synonyms
    "head pain": "headache",
    "head ache": "headache",
    "migraine": "headache",
    "cephalalgia": "headache",
    
    # Fatigue synonyms
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "lethargy": "fatigue",
    "weariness": "fatigue",
    
    # Nausea synonyms
    "queasiness": "nausea",
    "sick to stomach": "nausea",
    "upset stomach": "nausea",
    "nauseous": "nausea",
    
    # Respiratory synonyms
    "sob": "shortness of breath",
    "dyspnea": "shortness of breath",
    "dyspnoea": "shortness of breath",
    "breathlessness": "shortness of breath",
    "short of breath": "shortness of breath",
    "breathing difficulty": "difficulty breathing",
    "hard to breathe": "difficulty breathing",
    "trouble breathing": "difficulty breathing",
    
    # Throat synonyms
    "sorethroat": "sore throat",
    "pharyngitis": "sore throat",
    "throat infection": "sore throat",
    
    # Chest synonyms
    "chestpain": "chest pain",
    "angina": "chest pain",
    
    # Digestive synonyms
    "throwing up": "vomiting",
    "emesis": "vomiting",
    "vomit": "vomiting",
    "puking": "vomiting",
    "loose stools": "diarrhea",
    "diarrhoea": "diarrhea",
    "loose bowels": "diarrhea",
    "the runs": "diarrhea",
    "stomach ache": "abdominal pain",
    "belly ache": "abdominal pain",
    "tummy ache": "abdominal pain",
    "belly pain": "abdominal pain",
    
    # Dizziness synonyms
    "lightheadedness": "dizziness",
    "feeling faint": "dizziness",
    "wooziness": "dizziness",
    
    # Pain synonyms
    "backache": "back pain",
    "back ache": "back pain",
    "myalgia": "muscle pain",
    "muscle aches": "muscle pain",
    "body aches": "muscle pain",
    "arthralgia": "joint pain",
    "joint aches": "joint pain",
    
    # Skin synonyms
    "itchy rash": "rash",
    "skin eruption": "rash",
    "breakout": "rash",
    "contusions": "bruising",
    "ecchymosis": "bruising",
    
    # Urinary synonyms
    "urination burning": "burning urination",
    "painful urination": "burning urination",
    "dysuria": "burning urination",
    "frequent urination": "urinary frequency",
    "polyuria": "urinary frequency",
    
    # Nasal synonyms
    "runny nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "congestion": "nasal congestion",
    "blocked nose": "nasal congestion",
    
    # Cough synonyms
    "coughing": "cough",
    "tussis": "cough",
    
    # Sneeze synonyms
    "sneezes": "sneezing",
    "sneezing fits": "sneezing",
    
    # General synonyms
    "cold": "runny nose",
    "flu": "fever",
    "influenza": "fever",
    "weakness": "fatigue",
    "tired": "fatigue",
}

CANON = set(SYMPTOM_MAP.keys())

def _basic_clean(s: str) -> str:
    """Clean and normalize input text."""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _morph_reduce(t: str) -> str:
    """Naive stemming for common forms."""
    if t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    if t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    if t.endswith("es") and len(t) > 4:
        t = t[:-2]
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t

def normalize_token(tok: str) -> str:
    """Normalize a symptom token to canonical form."""
    t = _basic_clean(tok)
    
    # Remove common modifiers
    STOP = {
        "severe", "mild", "high", "low", "very", "slight", "acute", "chronic",
        "moderate", "persistent", "constant", "intermittent", "occasional",
        "frequent", "rare", "extreme", "intense", "dull", "sharp", "throbbing"
    }
    t = " ".join([w for w in t.split() if w not in STOP])
    
    # Direct synonym mapping
    if t in SYNONYMS:
        return SYNONYMS[t]
    
    # Try morphological reduction
    t_reduced = " ".join(_morph_reduce(w) for w in t.split())
    if t_reduced in SYNONYMS:
        return SYNONYMS[t_reduced]
    
    # Check if in canonical set
    if t in CANON:
        return t
    if t_reduced in CANON:
        return t_reduced
    
    # Fuzzy match to canonical keys
    match = difflib.get_close_matches(t, list(CANON), n=1, cutoff=0.85)
    if match:
        return match[0]
    
    match = difflib.get_close_matches(t_reduced, list(CANON), n=1, cutoff=0.85)
    return match[0] if match else t

def _pattern_expansions(t: str) -> list[str]:
    """Extract symptoms using pattern matching."""
    out = []
    
    # Pain in X → X pain
    pain_patterns = [
        (r"pain in (?:the |my )?(chest|heart)", "chest pain"),
        (r"pain in (?:the |my )?(back|spine)", "back pain"),
        (r"pain in (?:the |my )?(joint|joints)", "joint pain"),
        (r"pain in (?:the |my )?(muscle|muscles)", "muscle pain"),
        (r"pain in (?:the |my )?(abdomen|stomach|belly|gut)", "abdominal pain"),
        (r"pain in (?:the |my )?(head)", "headache"),
        (r"pain in (?:the |my )?(throat)", "throat pain"),
        (r"pain in (?:the |my )?(ear|ears)", "ear pain"),
        (r"pain in (?:the |my )?(eye|eyes)", "eye pain"),
        (r"pain in (?:the |my )?(neck)", "neck pain"),
        (r"pain in (?:the |my )?(shoulder|shoulders)", "shoulder pain"),
        (r"pain in (?:the |my )?(knee|knees)", "knee pain"),
        (r"pain in (?:the |my )?(hip|hips)", "hip pain"),
    ]
    
    for pattern, symptom in pain_patterns:
        if re.search(pattern, t):
            out.append(symptom)
    
    # Breathing difficulties
    breathing_patterns = [
        r"breathing difficulty",
        r"difficulty breathing",
        r"hard to breathe",
        r"trouble breathing",
        r"cant breathe",
        r"cannot breathe",
    ]
    for pattern in breathing_patterns:
        if re.search(pattern, t):
            out.append("shortness of breath")
            break
    
    # Sore throat
    if re.search(r"sore\s+throat", t):
        out.append("sore throat")
    
    # Running nose patterns
    if re.search(r"running nose|runny nose", t):
        out.append("runny nose")
    
    # Blood in various places
    if re.search(r"blood in (stool|urine|vomit|sputum)", t):
        match = re.search(r"blood in (stool|urine|vomit|sputum)", t)
        if match:
            location = match.group(1)
            if location == "stool":
                out.append("blood in stool")
            elif location == "urine":
                out.append("blood in urine")
            elif location == "vomit":
                out.append("vomiting blood")
    
    return out

def extract_symptoms(text: str) -> list[str]:
    """
    Extract symptoms from natural language text.
    
    Args:
        text: Input text describing symptoms
        
    Returns:
        List of canonical symptom names
    """
    if not text:
        return []
    
    t = _basic_clean(text)
    parts = [p.strip() for p in re.split(r",|;| and | with ", t) if p.strip()]
    out = []
    
    # Direct substring matches in cleaned text
    for name in CANON:
        if name in t:
            out.append(name)
    
    # Pattern-based expansions
    out.extend(_pattern_expansions(t))
    
    # Token normalization matches
    for p in parts:
        n = normalize_token(p)
        if n in CANON:
            out.append(n)
    
    return sorted(set(out))

def symptoms_to_vector(text: str, dim: int = 1300) -> np.ndarray:
    """
    Convert symptom text to vector representation.
    
    Args:
        text: Input text describing symptoms
        dim: Dimension of output vector (default 1300)
        
    Returns:
        Binary vector with 1s at symptom indices
    """
    vec = np.zeros(dim, dtype=np.float32)
    for name in extract_symptoms(text):
        idx = SYMPTOM_MAP.get(name)
        if idx is not None and 0 <= idx < dim:
            vec[idx] = 1
    return vec

def get_symptom_categories():
    """Return symptoms organized by body system."""
    return {
        "General/Constitutional": [k for k, v in SYMPTOM_MAP.items() if 100 <= v < 200],
        "Neurological": [k for k, v in SYMPTOM_MAP.items() if 200 <= v < 300],
        "Mental Health": [k for k, v in SYMPTOM_MAP.items() if 300 <= v < 350],
        "Head and Neck": [k for k, v in SYMPTOM_MAP.items() if 350 <= v < 400],
        "Respiratory": [k for k, v in SYMPTOM_MAP.items() if 400 <= v < 450],
        "Cardiovascular": [k for k, v in SYMPTOM_MAP.items() if 450 <= v < 500],
        "Gastrointestinal": [k for k, v in SYMPTOM_MAP.items() if 500 <= v < 600],
        "Hepatobiliary": [k for k, v in SYMPTOM_MAP.items() if 600 <= v < 620],
        "Urinary": [k for k, v in SYMPTOM_MAP.items() if 620 <= v < 670],
        "Reproductive - Female": [k for k, v in SYMPTOM_MAP.items() if 670 <= v < 700],
        "Reproductive - Male": [k for k, v in SYMPTOM_MAP.items() if 700 <= v < 720],
        "Musculoskeletal": [k for k, v in SYMPTOM_MAP.items() if 720 <= v < 800],
        "Skin": [k for k, v in SYMPTOM_MAP.items() if 800 <= v < 850],
        "Eye": [k for k, v in SYMPTOM_MAP.items() if 850 <= v < 880],
        "Hematologic": [k for k, v in SYMPTOM_MAP.items() if 880 <= v < 900],
        "Allergy/Immune": [k for k, v in SYMPTOM_MAP.items() if 900 <= v < 930],
        "Endocrine": [k for k, v in SYMPTOM_MAP.items() if 930 <= v < 960],
        "Pregnancy": [k for k, v in SYMPTOM_MAP.items() if 960 <= v < 980],
        "Infectious": [k for k, v in SYMPTOM_MAP.items() if 980 <= v < 1000],
        "Trauma": [k for k, v in SYMPTOM_MAP.items() if 1000 <= v < 1020],
        "Systemic/Other": [k for k, v in SYMPTOM_MAP.items() if 1020 <= v < 1100],
    }

# Example usage and testing
if __name__ == "__main__":
    # Test examples
    test_cases = [
        "I have fever, headache, and sore throat",
        "Patient presents with severe chest pain and shortness of breath",
        "Experiencing nausea, vomiting, and abdominal pain",
        "Pain in my back and difficulty walking",
        "Running nose, sneezing, and coughing",
        "Blood in urine and burning when urinating",
    ]
    
    print(f"Total symptoms in map: {len(SYMPTOM_MAP)}")
    print(f"\nTesting symptom extraction:\n")
    
    for test in test_cases:
        symptoms = extract_symptoms(test)
        print(f"Input: {test}")
        print(f"Extracted: {symptoms}")
        print()