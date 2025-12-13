"""
Medical Term Dictionary for M3D-LaMed

Contains multiple descriptions for medical terms, used for:
- Referring expression segmentation
- Definition-based queries
- Diverse prompt generation

Based on the M3D-LaMed paper's term dictionary approach (Appendix, Figure 10).
"""

from typing import Dict, List, Optional
import random


# =============================================================================
# ORGAN DESCRIPTIONS
# =============================================================================

ORGAN_TERMS: Dict[str, List[str]] = {
    # Abdominal organs
    "liver": [
        "Primary organ responsible for detoxifying the blood by removing harmful substances.",
        "Produces bile, a fluid that aids in the digestion and absorption of fats.",
        "Large organ in the upper right abdomen with various metabolic functions.",
        "Largest internal organ, essential for metabolism and detoxification.",
        "Organ that stores and regulates glycogen, a crucial energy reserve.",
    ],
    "spleen": [
        "Organ involved in blood filtration and immune response.",
        "Lymphoid organ that filters blood and removes old red blood cells.",
        "Organ in the left upper abdomen that plays a role in immunity.",
        "Blood-filtering organ located in the left hypochondrium.",
    ],
    "pancreas": [
        "Glandular organ located behind the stomach in the abdomen.",
        "Organ that produces insulin and digestive enzymes.",
        "Retroperitoneal organ with endocrine and exocrine functions.",
        "Gland responsible for blood sugar regulation and digestion.",
    ],
    "kidney": [
        "Pair of bean-shaped organs involved in waste excretion and fluid balance.",
        "Organs crucial for filtering blood and producing urine.",
        "Bean-shaped organs integral to waste removal and urine production.",
        "Pair of organs responsible for filtering waste from the blood.",
        "Organs vital for removing toxins and excess fluids from the body.",
    ],
    "left kidney": [
        "Kidney located on the left side of the retroperitoneum.",
        "Left-sided bean-shaped organ for blood filtration.",
        "Renal organ on the left side responsible for urine production.",
    ],
    "right kidney": [
        "Kidney located on the right side of the retroperitoneum.",
        "Right-sided bean-shaped organ for blood filtration.",
        "Renal organ on the right side responsible for urine production.",
    ],
    "gallbladder": [
        "Small organ that stores bile produced by the liver.",
        "Pear-shaped organ beneath the liver storing digestive bile.",
        "Organ that concentrates and stores bile for fat digestion.",
    ],
    "stomach": [
        "Digestive organ that breaks down food with acid and enzymes.",
        "Muscular organ of the digestive tract receiving food from the oesophagus.",
        "J-shaped organ responsible for initial food digestion.",
    ],
    "small bowel": [
        "Long tubular organ where most nutrient absorption occurs.",
        "Intestinal segment comprising duodenum, jejunum, and ileum.",
        "Primary site of nutrient absorption in the digestive tract.",
    ],
    "large bowel": [
        "Final part of the digestive tract absorbing water and forming stool.",
        "Colon and rectum, responsible for water absorption and waste storage.",
        "Large intestine including ascending, transverse, descending colon.",
    ],
    "bladder": [
        "Organ in the lower abdomen storing urine before excretion.",
        "Muscular sac that stores urine from the kidneys.",
        "Hollow organ in the pelvis for urine storage.",
    ],

    # Thoracic organs
    "heart": [
        "Organ responsible for pumping blood throughout the body.",
        "Muscular organ that circulates blood through the cardiovascular system.",
        "Central pump of the circulatory system.",
        "Vital organ that pumps oxygenated blood to tissues and organs.",
    ],
    "lung": [
        "Respiratory organ responsible for gas exchange.",
        "Pair of organs in the thoracic cavity for breathing.",
        "Organs that transfer oxygen to blood and remove carbon dioxide.",
    ],
    "left lung": [
        "Lung located on the left side of the chest.",
        "Respiratory organ in the left thoracic cavity.",
        "Left-sided pulmonary organ with two lobes.",
    ],
    "right lung": [
        "Lung located on the right side of the chest.",
        "Respiratory organ in the right thoracic cavity.",
        "Right-sided pulmonary organ with three lobes.",
    ],
    "trachea": [
        "Airway connecting the larynx to the bronchi.",
        "Windpipe carrying air to and from the lungs.",
        "Tubular structure enabling air passage to the lungs.",
    ],
    "oesophagus": [
        "Muscular tube connecting the throat to the stomach.",
        "Digestive tract segment transporting food to the stomach.",
        "Tubular organ for food passage from pharynx to stomach.",
    ],
    "aorta": [
        "Largest artery in the body carrying blood from the heart.",
        "Main arterial trunk distributing oxygenated blood.",
        "Primary artery originating from the left ventricle.",
    ],

    # Head and neck structures
    "brain": [
        "Central organ of the nervous system controlling body functions.",
        "Organ responsible for thought, memory, and bodily control.",
        "Complex organ in the cranium controlling cognition and movement.",
    ],
    "thyroid": [
        "Butterfly-shaped gland in the neck regulating metabolism.",
        "Endocrine gland producing thyroid hormones.",
        "Gland in the anterior neck controlling metabolic rate.",
    ],
    "parathyroid": [
        "Small glands behind the thyroid regulating calcium.",
        "Endocrine glands controlling calcium and phosphorus levels.",
    ],

    # Vascular structures
    "portal vein": [
        "Vein carrying blood from the digestive organs to the liver.",
        "Major vessel draining blood from the gastrointestinal tract.",
    ],
    "inferior vena cava": [
        "Large vein returning blood from lower body to the heart.",
        "Major venous trunk draining blood from below the diaphragm.",
    ],

    # Skeletal structures
    "vertebrae": [
        "Bones of the spinal column protecting the spinal cord.",
        "Individual segments of the spine providing support and flexibility.",
    ],
    "ribs": [
        "Curved bones forming the thoracic cage.",
        "Bony structures protecting the heart and lungs.",
    ],
}


# =============================================================================
# PATHOLOGY/ABNORMALITY DESCRIPTIONS
# =============================================================================

PATHOLOGY_TERMS: Dict[str, List[str]] = {
    # Masses and tumours
    "tumour": [
        "Abnormal growth in tissue.",
        "Mass of cells forming abnormally.",
        "Neoplastic lesion.",
        "Uncontrolled cell proliferation forming a mass.",
        "An irregular tissue growth characterised by rapid division.",
    ],
    "mass": [
        "Abnormal collection of tissue.",
        "Localised swelling or growth.",
        "Space-occupying lesion.",
    ],
    "nodule": [
        "Small rounded mass of tissue.",
        "Focal lesion typically less than 3cm.",
        "Circumscribed rounded abnormality.",
    ],
    "cyst": [
        "Fluid-filled sac within tissue.",
        "Enclosed cavity containing liquid material.",
        "Benign fluid collection with defined walls.",
    ],

    # Vascular abnormalities
    "haemorrhage": [
        "Bleeding into tissue or body cavity.",
        "Extravasation of blood from vessels.",
        "Accumulation of blood outside vessels.",
    ],
    "thrombosis": [
        "Blood clot within a vessel.",
        "Occlusion of a vessel by clotted blood.",
        "Intravascular blood coagulation.",
    ],
    "aneurysm": [
        "Abnormal dilation of a blood vessel.",
        "Localised widening of an artery.",
        "Vessel wall weakening causing bulging.",
    ],

    # Inflammatory conditions
    "inflammation": [
        "Tissue response to injury or infection.",
        "Swelling and redness indicating immune response.",
        "Protective tissue reaction to harmful stimuli.",
    ],
    "abscess": [
        "Collection of pus within tissue.",
        "Localised infection with purulent material.",
        "Walled-off collection of infected material.",
    ],
    "consolidation": [
        "Lung tissue filled with fluid or cellular material.",
        "Replacement of air in lung with pathological material.",
        "Dense opacity in lung parenchyma.",
    ],

    # Fluid collections
    "effusion": [
        "Abnormal fluid collection in a body cavity.",
        "Accumulation of fluid in potential space.",
        "Fluid in pleural, pericardial, or peritoneal space.",
    ],
    "ascites": [
        "Free fluid in the peritoneal cavity.",
        "Abdominal fluid accumulation.",
        "Fluid collection in the abdomen.",
    ],
    "oedema": [
        "Swelling caused by excess fluid in tissues.",
        "Tissue fluid accumulation.",
        "Abnormal interstitial fluid collection.",
    ],

    # Degenerative changes
    "calcification": [
        "Deposition of calcium in tissue.",
        "Calcium accumulation in soft tissue.",
        "Mineralisation of tissue.",
    ],
    "atrophy": [
        "Reduction in tissue size or volume.",
        "Shrinkage of an organ or tissue.",
        "Decrease in cell size or number.",
    ],

    # Structural abnormalities
    "stenosis": [
        "Abnormal narrowing of a passage or vessel.",
        "Constriction of a tubular structure.",
        "Pathological narrowing.",
    ],
    "obstruction": [
        "Blockage of a passage or structure.",
        "Impediment to normal flow or passage.",
        "Occlusion of a tubular structure.",
    ],
    "perforation": [
        "Hole or rupture in tissue or organ wall.",
        "Breach in the wall of a hollow organ.",
        "Abnormal opening through tissue.",
    ],
    "fracture": [
        "Break in bone continuity.",
        "Disruption of bony structure.",
        "Skeletal discontinuity.",
    ],
}


# =============================================================================
# ANATOMICAL LOCATION TERMS
# =============================================================================

LOCATION_TERMS: Dict[str, List[str]] = {
    "left": ["On the left side", "Left-sided", "Sinister"],
    "right": ["On the right side", "Right-sided", "Dexter"],
    "bilateral": ["On both sides", "Affecting both sides", "Bilateral distribution"],
    "midline": ["In the centre", "Along the midline", "Central"],
    "anterior": ["In front", "Towards the front", "Ventral"],
    "posterior": ["Behind", "Towards the back", "Dorsal"],
    "superior": ["Above", "Upper", "Cranial"],
    "inferior": ["Below", "Lower", "Caudal"],
    "medial": ["Towards the midline", "Inner", "Central"],
    "lateral": ["Away from midline", "Outer", "Peripheral"],
    "peripheral": ["At the edges", "Away from centre", "Marginal"],
    "central": ["In the centre", "Core", "Middle"],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_organ_description(organ: str) -> str:
    """Get a random description for an organ."""
    organ_lower = organ.lower()
    if organ_lower in ORGAN_TERMS:
        return random.choice(ORGAN_TERMS[organ_lower])
    return f"Anatomical structure: {organ}"


def get_pathology_description(pathology: str) -> str:
    """Get a random description for a pathology term."""
    pathology_lower = pathology.lower()
    if pathology_lower in PATHOLOGY_TERMS:
        return random.choice(PATHOLOGY_TERMS[pathology_lower])
    return f"Pathological finding: {pathology}"


def get_all_descriptions(term: str) -> List[str]:
    """Get all descriptions for a term (organ or pathology)."""
    term_lower = term.lower()
    if term_lower in ORGAN_TERMS:
        return ORGAN_TERMS[term_lower]
    if term_lower in PATHOLOGY_TERMS:
        return PATHOLOGY_TERMS[term_lower]
    return [f"Medical term: {term}"]


def get_definition_prompt(term: str) -> str:
    """
    Generate a definition-based prompt for a medical term.

    Args:
        term: Medical term (organ or pathology)

    Returns:
        Formatted definition prompt for referring expression tasks
    """
    description = get_organ_description(term) if term.lower() in ORGAN_TERMS else get_pathology_description(term)
    templates = [
        f"Description: {description} Please identify and describe it.",
        f"Defining it as: {description} What is this structure?",
        f"Given the definition: {description} Identify this in the image.",
    ]
    return random.choice(templates)


def list_available_terms() -> Dict[str, List[str]]:
    """List all available terms in the dictionary."""
    return {
        "organs": list(ORGAN_TERMS.keys()),
        "pathologies": list(PATHOLOGY_TERMS.keys()),
        "locations": list(LOCATION_TERMS.keys()),
    }
