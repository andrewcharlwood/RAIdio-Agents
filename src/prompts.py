"""
Prompt Templates for M3D-LaMed

Based on the M3D-LaMed paper (arXiv:2404.00578v1) prompting guide.
Implements templates for VQA, report generation, and analysis tasks.
"""

from typing import Dict, List, Optional, Tuple
import random


# =============================================================================
# REPORT GENERATION PROMPTS
# =============================================================================

REPORT_GENERATION_PROMPTS = [
    "Describe the findings of the medical image you see.",
    "What are the findings presented in this medical scan?",
    "Can you provide a caption consisting of findings for this medical image?",
    "Please write a caption consisting of findings for this scan.",
    "Can you summarise with findings the images presented?",
    "What is the findings of this image?",
    "Describe this medical scan with findings.",
    "Please caption this medical scan with findings.",
]


# =============================================================================
# VQA PROMPT TEMPLATES BY CATEGORY
# =============================================================================

# Plane identification (98.80% accuracy - highest)
PLANE_QUESTIONS = {
    "closed": [
        "Which plane is displayed in the image? Choices: A. Axial B. Sagittal C. Coronal D. Oblique",
        "What plane is this image in? Choices: A. Axial B. Sagittal C. Coronal D. Oblique",
        "In which anatomical plane is this scan acquired? Choices: A. Axial B. Sagittal C. Coronal D. Oblique",
    ],
    "open": [
        "Which plane is displayed in the image?",
        "What plane is this image in?",
        "In which anatomical plane is this scan acquired?",
    ],
}

# CT Phase identification (79.75% accuracy)
PHASE_QUESTIONS = {
    "closed": [
        "What is the CT phase shown? Choices: A. Non-contrast B. Arterial phase C. Portal venous phase D. Delayed phase",
        "What CT phase is this image? Choices: A. Non-contrast B. Contrast C. Arterial phase D. Portal venous phase",
        "Which contrast phase is demonstrated? Choices: A. Non-contrast B. Arterial C. Portal venous D. Venous phase",
    ],
    "open": [
        "What is the CT phase shown in this image?",
        "What phase of contrast enhancement is this?",
        "Is this a contrast or non-contrast study?",
    ],
}

# Organ identification (74.75% accuracy)
ORGAN_QUESTIONS = {
    "closed_templates": [
        "Which organ shows abnormality in the image? Choices: A. {a} B. {b} C. {c} D. {d}",
        "What organ is affected? Choices: A. {a} B. {b} C. {c} D. {d}",
        "Which organ demonstrates the finding? Choices: A. {a} B. {b} C. {c} D. {d}",
    ],
    "open": [
        "Which organs are visible in this image?",
        "What anatomical structures can you identify?",
        "Describe the organs visible in this scan.",
    ],
}

# Abnormality identification (66.65% accuracy)
ABNORMALITY_QUESTIONS = {
    "closed_templates": [
        "What type of abnormality can be observed? Choices: A. {a} B. {b} C. {c} D. {d}",
        "What is the nature of the anomaly found? Choices: A. {a} B. {b} C. {c} D. {d}",
        "Which abnormality is present? Choices: A. {a} B. {b} C. {c} D. {d}",
    ],
    "open": [
        "What abnormality is present in this scan?",
        "Identify any abnormalities or pathological findings visible.",
        "What type of abnormality can be observed in the image?",
        "Describe any pathological findings you can identify.",
    ],
    "word_limit": [
        "Please use a word to describe the abnormality.",
        "Please use three words to describe the abnormality.",
        "Please use five words to describe the abnormality.",
    ],
}

# Location identification (58.94% accuracy - lowest)
LOCATION_QUESTIONS = {
    "closed_templates": [
        "Where is the abnormality located? Choices: A. {a} B. {b} C. {c} D. {d}",
        "In which region is the finding present? Choices: A. {a} B. {b} C. {c} D. {d}",
        "Which side is affected? Choices: A. Left B. Right C. Bilateral D. Midline",
    ],
    "open": [
        "Where is the abnormality located?",
        "In which region of the anatomy is the finding present?",
        "Describe the location of any findings.",
    ],
}


# =============================================================================
# ANALYSIS CHAIN TEMPLATES
# =============================================================================

def get_analysis_chain(modality: str = "CT") -> List[Tuple[str, str, str]]:
    """
    Get ordered analysis chain for comprehensive scan analysis.

    Based on paper recommendation: start with high-accuracy queries (plane, phase)
    before moving to lower-accuracy queries (abnormality, location).

    Returns:
        List of (name, question, question_type) tuples
    """
    chain = [
        # Step 1: Establish image parameters (high accuracy)
        ("plane", "What plane is this image in?", "open"),

        # Step 2: Identify contrast phase (high accuracy, CT only)
        ("phase", f"What is the {modality} phase shown in this image?", "open"),

        # Step 3: Identify visible structures
        ("structures", "What anatomical structures are visible in this scan?", "open"),

        # Step 4: Screen for abnormalities (open-ended to avoid leading)
        ("abnormalities",
         "Identify any abnormalities or pathological findings visible in this scan. "
         "Describe their appearance, location, and clinical significance. "
         "If no abnormalities are present, state that clearly.",
         "open"),

        # Step 5: Key clinical findings
        ("key_findings",
         "What are the most clinically significant findings in this image? "
         "List them in order of importance with confidence levels.",
         "open"),

        # Step 6: Differential diagnosis
        ("differential",
         "Based on the findings in this scan, what differential diagnoses would you consider? "
         "Provide reasoning for each.",
         "open"),
    ]

    return chain


def get_quick_analysis_chain(modality: str = "CT") -> List[Tuple[str, str, str]]:
    """
    Get shortened analysis chain for faster processing.

    Returns:
        List of (name, question, question_type) tuples
    """
    return [
        ("overview",
         f"Provide a comprehensive analysis of this {modality} scan. "
         "Describe key anatomical structures visible, any abnormalities, "
         "and overall image quality.",
         "open"),

        ("findings",
         "What are the key clinical findings? List any abnormalities with "
         "their location and clinical significance.",
         "open"),

        ("assessment",
         "Based on your analysis, what is your overall assessment and "
         "what follow-up would you recommend?",
         "open"),
    ]


# =============================================================================
# CLOSED-ENDED VQA FOR SPECIFIC PATHOLOGIES
# =============================================================================

# Head/Neck pathology screening questions
HEAD_NECK_PATHOLOGY_QUESTIONS = {
    "mastoiditis": {
        "question": "Is there evidence of mastoiditis in the temporal bones?",
        "choices": ["A. Yes, definite mastoiditis", "B. Possible/subtle findings",
                   "C. No evidence of mastoiditis", "D. Cannot assess"],
    },
    "sinusitis": {
        "question": "Is there evidence of sinusitis?",
        "choices": ["A. Yes, with air-fluid levels", "B. Mucosal thickening only",
                   "C. No evidence of sinusitis", "D. Cannot assess"],
    },
    "intracranial_mass": {
        "question": "Is there evidence of an intracranial mass or lesion?",
        "choices": ["A. Yes, mass lesion present", "B. Possible/indeterminate",
                   "C. No mass lesion", "D. Cannot assess"],
    },
    "haemorrhage": {
        "question": "Is there evidence of intracranial haemorrhage?",
        "choices": ["A. Yes, acute haemorrhage", "B. Subacute/chronic blood products",
                   "C. No haemorrhage", "D. Cannot assess"],
    },
    "hydrocephalus": {
        "question": "Is there evidence of hydrocephalus or ventricular enlargement?",
        "choices": ["A. Yes, hydrocephalus present", "B. Borderline ventricular size",
                   "C. Normal ventricles", "D. Cannot assess"],
    },
    "midline_shift": {
        "question": "Is there midline shift?",
        "choices": ["A. Yes, significant shift", "B. Mild shift",
                   "C. No midline shift", "D. Cannot assess"],
    },
}

# Abdominal pathology screening questions
ABDOMINAL_PATHOLOGY_QUESTIONS = {
    "liver_lesion": {
        "question": "Is there a focal liver lesion?",
        "choices": ["A. Yes, solid mass", "B. Yes, cystic lesion",
                   "C. No focal lesion", "D. Cannot assess"],
    },
    "kidney_lesion": {
        "question": "Is there a renal abnormality?",
        "choices": ["A. Yes, mass/tumour", "B. Yes, cyst or stone",
                   "C. No renal abnormality", "D. Cannot assess"],
    },
    "bowel_obstruction": {
        "question": "Is there evidence of bowel obstruction?",
        "choices": ["A. Yes, small bowel obstruction", "B. Yes, large bowel obstruction",
                   "C. No obstruction", "D. Cannot assess"],
    },
    "free_fluid": {
        "question": "Is there free fluid in the abdomen?",
        "choices": ["A. Yes, significant ascites", "B. Trace free fluid",
                   "C. No free fluid", "D. Cannot assess"],
    },
}

# Chest pathology screening questions
CHEST_PATHOLOGY_QUESTIONS = {
    "lung_nodule": {
        "question": "Is there a pulmonary nodule or mass?",
        "choices": ["A. Yes, nodule present", "B. Yes, mass present",
                   "C. No nodule or mass", "D. Cannot assess"],
    },
    "consolidation": {
        "question": "Is there pulmonary consolidation or ground glass opacity?",
        "choices": ["A. Yes, consolidation", "B. Yes, ground glass",
                   "C. No parenchymal abnormality", "D. Cannot assess"],
    },
    "pleural_effusion": {
        "question": "Is there pleural effusion?",
        "choices": ["A. Yes, large effusion", "B. Yes, small effusion",
                   "C. No pleural effusion", "D. Cannot assess"],
    },
    "pneumothorax": {
        "question": "Is there pneumothorax?",
        "choices": ["A. Yes, tension pneumothorax", "B. Yes, simple pneumothorax",
                   "C. No pneumothorax", "D. Cannot assess"],
    },
}


def get_pathology_screen(region: str = "head") -> Dict:
    """
    Get appropriate pathology screening questions for body region.

    Args:
        region: "head", "chest", or "abdomen"

    Returns:
        Dictionary of pathology screening questions
    """
    screens = {
        "head": HEAD_NECK_PATHOLOGY_QUESTIONS,
        "neck": HEAD_NECK_PATHOLOGY_QUESTIONS,
        "chest": CHEST_PATHOLOGY_QUESTIONS,
        "thorax": CHEST_PATHOLOGY_QUESTIONS,
        "abdomen": ABDOMINAL_PATHOLOGY_QUESTIONS,
        "pelvis": ABDOMINAL_PATHOLOGY_QUESTIONS,
    }
    return screens.get(region.lower(), HEAD_NECK_PATHOLOGY_QUESTIONS)


def format_closed_question(question: str, choices: List[str]) -> str:
    """Format a closed-ended question with choices."""
    choices_str = " ".join(choices)
    return f"{question} Choices: {choices_str}"


# =============================================================================
# SEGMENTATION PROMPTS (for future use)
# =============================================================================

SEMANTIC_SEGMENTATION_PROMPTS = [
    "Can you segment the {target} in this image?",
    "Can you segment {target} in this image? Please output the mask.",
    "Please segment the {target} in this image.",
    "Could you provide a segmentation for the {target}?",
]

REFERRING_SEGMENTATION_PROMPTS = [
    "Description: {description} Please segment it.",
    "Defining it as: {definition} Now, segment and provide your answer.",
    "The description provided is: {description} Now, segment it and provide your answer.",
    "Based on the provided definition: {definition} Please segment and provide your response.",
]


# =============================================================================
# POSITIONING PROMPTS (for future use)
# =============================================================================

REFERRING_COMPREHENSION_PROMPTS = [
    "Can you find the {target} in this image? Give coordinates.",
    "Where is {target} in this image? Please output the box.",
    "Please bounding the {target} by box in this image.",
]

REFERRING_GENERATION_PROMPTS = [
    "What target is present within the coordinates {coords}?",
    "What is the area marked with a box {coords} in the image? Can you explain it?",
    "Please describe the target and its function based on the box {coords} in the image.",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_random_prompt(prompt_list: List[str]) -> str:
    """Get a random prompt from a list."""
    return random.choice(prompt_list)


def build_analysis_prompt(
    question: str,
    modality: str = "CT",
    include_confidence: bool = True,
    include_location: bool = True,
) -> str:
    """
    Build a comprehensive analysis prompt with optional modifiers.

    Args:
        question: Base question
        modality: CT or MRI
        include_confidence: Add request for confidence levels
        include_location: Add request for anatomical locations

    Returns:
        Enhanced prompt string
    """
    prompt = question

    additions = []
    if include_location:
        additions.append("Be specific about anatomical location.")
    if include_confidence:
        additions.append("Indicate your confidence level for each finding.")

    if additions:
        prompt = f"{prompt} {' '.join(additions)}"

    return prompt


# =============================================================================
# DEFAULT SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_GENERAL = (
    "You are a medical imaging AI assistant. Analyse the provided 3D medical image "
    "and answer questions accurately. If you cannot determine something with "
    "confidence, say so."
)

SYSTEM_PROMPT_RADIOLOGIST = (
    "You are an expert radiologist AI assistant analysing 3D medical images. "
    "Provide detailed, clinically relevant findings. Use appropriate medical "
    "terminology. If findings are uncertain, indicate the level of confidence. "
    "Always consider differential diagnoses for abnormal findings."
)

SYSTEM_PROMPT_SCREENING = (
    "You are a medical imaging AI performing initial screening of 3D scans. "
    "Focus on identifying significant abnormalities that require clinical attention. "
    "Be thorough but concise. Flag any urgent findings clearly."
)
