"""
Example worker script: generate Q&A using different persona configurations.

This script demonstrates how to define persona configurations using
`opengovcorpus.define_config` and run dataset creation iteratively for
multiple persona variants. For testing we limit scraping with `max_pages=1`.

Note: Running create_dataset will perform scraping and (optionally) call a
local HF model or the environment-configured model. Use small `max_pages`
when trying different persona values.
"""

from pathlib import Path
import sys

# Ensure the project root is on sys.path so the local opengovcorpus package is imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opengovcorpus as og

# Example page to scrape for tests (small browse page)
TEST_URL = "https://www.gov.uk/browse/benefits"

# Define several persona variants we want to iterate over
personas = [
    og.define_config(
        targetAgeGroup="18-25",
        genderIdentity="female",
        educationBackground="secondary",
        targetProfession="retail",
        digitalLiteracy="medium",
        geoRegion="England",
        householdIncomeStatus="moderate",
        targetRole="service-user",
        promptIntentType="informational",
    ),
    og.define_config(
        targetAgeGroup="46-65",
        genderIdentity="male",
        educationBackground="tertiary",
        targetProfession="teacher",
        digitalLiteracy="high",
        geoRegion="Scotland",
        householdIncomeStatus="above moderate",
        targetRole="service-user",
        promptIntentType="procedural",
    ),
]

for i, p in enumerate(personas, start=1):
    name = f"gov-qa-persona-{i}"
    print(f"\n=== Running persona {i} -> {name} ===")

    # Limit pages for testing; set max_pages to None for full scrape
    og.create_dataset(
        name=name,
        url=TEST_URL,
        include_metadata=True,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        max_pages=1,
        persona_config=p,
    )

print("\nAll persona runs complete. Check the OpenGovCorpus-* folders for outputs.")
