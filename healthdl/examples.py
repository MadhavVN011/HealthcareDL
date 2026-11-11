"""Fictional walkthrough inputs, separate from a visitor's own symptom list."""

DEMO_EXAMPLES = (
    {"id": "skin", "label": "Skin & itching", "symptoms": (
        "itching", "skin_rash", "nodal_skin_eruptions"), "expected_id": "fungal_infection"},
    {"id": "blemishes", "label": "Skin blemishes", "symptoms": (
        "pus_filled_pimples", "blackheads", "scurring"), "expected_id": "acne"},
    {"id": "sneezing", "label": "Sneezing & chills", "symptoms": (
        "continuous_sneezing", "shivering", "chills", "watering_from_eyes"), "expected_id": "allergy"},
    {"id": "blisters", "label": "Blister pattern", "symptoms": (
        "blister", "red_sore_around_nose", "yellow_crust_ooze"), "expected_id": "impetigo"},
)
