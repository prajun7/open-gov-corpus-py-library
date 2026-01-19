"""
Generate one simple and one complex Q&A per persona profile for gov.uk.

This script demonstrates:
- Creating persona profiles for all age groups via opengovcorpus.define_config
- Scraping a small number of pages from https://www.gov.uk/ (set max_pages small for tests)
- For each profile, generating exactly two Q&A pairs (simple + complex)
- Writing a combined CSV at usage_examples/outputs/gov_uk_profiles_qas.csv

Notes:
- If a local HF/text-generation pipeline is available (transformers + torch), the script will try to use it
  via the library's DatasetCreator._generator. Otherwise it falls back to a simple heuristic generator.
- The output CSV intentionally omits any "promptIntentType"/"reasoningComplexity" fields (per request).

Run:
    python3 usage_examples/generate_simple_complex_profiles.py

"""
import os
import csv
import json
from pathlib import Path
from datetime import datetime
import sys

# Ensure the project root is on sys.path so the local opengovcorpus package is imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opengovcorpus as og
from opengovcorpus.scraper import scrape_website
from opengovcorpus.models import DatasetConfig
from opengovcorpus.dataset import DatasetCreator, _parse_provenance_from_text, FINAL_INSTRUCTION_PROMPT

# Output path
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "gov_uk_profiles_qas.csv"

# Target URL (top-level gov.uk)
TARGET_URL = "https://www.gov.uk/"

# Age groups and example persona defaults
AGE_GROUPS = ["under18", "18-25", "26-45", "46-65", "65+"]

# Create simple profile defaults per age group
def build_profiles():
    profiles = []
    for age in AGE_GROUPS:
        # Make plausible default values per age group
        if age == "under18":
            p = og.define_config(
                targetAgeGroup=age,
                genderIdentity="unspecified",
                educationBackground="secondary",
                targetProfession="student",
                digitalLiteracy="medium",
                geoRegion="England",
                householdIncomeStatus="moderate",
                targetRole="service-user",
                promptIntentType="informational",
            )
        elif age == "18-25":
            p = og.define_config(
                targetAgeGroup=age,
                genderIdentity="non-binary",
                educationBackground="secondary",
                targetProfession="retail",
                digitalLiteracy="high",
                geoRegion="England",
                householdIncomeStatus="moderate",
                targetRole="service-user",
                promptIntentType="informational",
            )
        elif age == "26-45":
            p = og.define_config(
                targetAgeGroup=age,
                genderIdentity="female",
                educationBackground="tertiary",
                targetProfession="nurse",
                digitalLiteracy="high",
                geoRegion="England",
                householdIncomeStatus="above moderate",
                targetRole="service-user",
                promptIntentType="procedural",
            )
        elif age == "46-65":
            p = og.define_config(
                targetAgeGroup=age,
                genderIdentity="male",
                educationBackground="tertiary",
                targetProfession="teacher",
                digitalLiteracy="medium",
                geoRegion="Scotland",
                householdIncomeStatus="above moderate",
                targetRole="service-user",
                promptIntentType="procedural",
            )
        else:  # 65+
            p = og.define_config(
                targetAgeGroup=age,
                genderIdentity="unspecified",
                educationBackground="secondary",
                targetProfession="retired",
                digitalLiteracy="low",
                geoRegion="Wales",
                householdIncomeStatus="moderate",
                targetRole="service-user",
                promptIntentType="informational",
            )
        profiles.append(p)
    return profiles


def generate_for_profile(profile: dict, max_pages: int = 1):
    """Scrape and generate exactly two Q&A pairs for the given persona profile.

    Returns a list of dict rows ready to be written to CSV.
    """
    rows = []

    # Minimal DatasetConfig for initializing DatasetCreator (so we can reuse HF generator if available)
    cfg = DatasetConfig(name=f"tmp-{profile['targetAgeGroup']}", url=TARGET_URL, max_pages=max_pages, persona_config=profile)
    creator = DatasetCreator(cfg)

    # Scrape pages (this may be slow; keep max_pages small during testing)
    scraped = scrape_website(TARGET_URL, max_pages=max_pages)
    if not scraped:
        print(f"No content scraped for {TARGET_URL}")
        return rows

    # Use the first scraped item to generate Q&A pairs
    for item in scraped:
        # Build a single input_text block (use content)
        input_text = item.content

        # Build a custom prompt asking for exactly 2 Q&A: 1 simple, 1 complex,
        # and explicitly request output JSON array where each item contains the required fields.
        persona_json = json.dumps(profile, indent=2)
        custom_instruction = (
            "Generate exactly 2 Q&A items from the INPUT TEXT: one SIMPLE question (short, direct), "
            "and one COMPLEX question (multi-step or higher reasoning).\n"
            "Use ONLY the INPUT TEXT for answers. Output a JSON array. For each item include these keys: \n"
            "  prompt, response, confidenceScore, serviceDomain, subServiceDomain, topic, sourceURL, sourceDomain, "
            "sourceLicense, documentType, dateCreated, language\n"
            "Do NOT include promptIntentType or reasoningComplexity fields in the output.\n\n"
            "PERSONA:\n" + persona_json + "\n\n"
            "INPUT TEXT:\n" + input_text
        )

        # Try to use the initialized generator if present
        generated_items = []
        gen = getattr(creator, "_generator", None)
        if creator.use_llm and gen is not None:
            try:
                outputs = gen(
                    custom_instruction,
                    max_new_tokens=creator.max_new_tokens,
                    do_sample=creator.do_sample,
                    temperature=creator.temperature,
                )
                text = outputs[0].get("generated_text", "") if outputs else ""
                # Try to extract JSON
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    json_str = text[start:end+1]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        generated_items = parsed
            except Exception as e:
                print("LLM generation failed, falling back to heuristic:", e)
                generated_items = []

        # Fallback heuristic if no LLM output
        if not generated_items:
            # Simple: take the first sentence as answer and craft a direct question
            first_sent = input_text.split(". ")
            s_answer = first_sent[0].strip() if first_sent else ""
            simple_prompt = f"What does the text say about '{s_answer[:80]}'?"
            simple_resp = s_answer

            # Complex: ask a how/step question referencing multiple sentences
            complex_context = " ".join(first_sent[:3]).strip()
            complex_prompt = f"How would I follow steps described here: '{complex_context[:120]}'?"
            complex_resp = complex_context

            # Build items (matching the requested keys)
            now = datetime.now().strftime("%Y-%m-%d")
            base_prov = _parse_provenance_from_text(input_text)
            serviceDomain, subServiceDomain, topic, _, browse_suffix = base_prov
            sourceURL = f"https://www.gov.uk/browse/{browse_suffix}" if serviceDomain != "Unknown" else item.url

            generated_items = [
                {
                    "prompt": simple_prompt,
                    "response": simple_resp,
                    "confidenceScore": float(profile.get("confidenceScore", 1.0)),
                    "serviceDomain": serviceDomain,
                    "subServiceDomain": subServiceDomain,
                    "topic": topic,
                    "sourceURL": sourceURL,
                    "sourceDomain": "www.gov.uk",
                    "sourceLicense": "Open Government Licence (OGL) v3.0",
                    "documentType": "webpage",
                    "dateCreated": now,
                    "language": "en",
                },
                {
                    "prompt": complex_prompt,
                    "response": complex_resp,
                    "confidenceScore": float(profile.get("confidenceScore", 1.0)),
                    "serviceDomain": serviceDomain,
                    "subServiceDomain": subServiceDomain,
                    "topic": topic,
                    "sourceURL": sourceURL,
                    "sourceDomain": "www.gov.uk",
                    "sourceLicense": "Open Government Licence (OGL) v3.0",
                    "documentType": "webpage",
                    "dateCreated": now,
                    "language": "en",
                },
            ]

        # Normalize items and attach persona fields into row (explicit fields, excluding promptIntentType)
        for it in generated_items:
            row = {
                "prompt": it.get("prompt"),
                "response": it.get("response"),
                "targetAgeGroup": profile.get("targetAgeGroup"),
                "genderIdentity": profile.get("genderIdentity"),
                "educationBackground": profile.get("educationBackground"),
                "targetProfession": profile.get("targetProfession"),
                "digitalLiteracy": profile.get("digitalLiteracy"),
                "geoRegion": profile.get("geoRegion"),
                "householdIncomeStatus": profile.get("householdIncomeStatus"),
                "targetRole": profile.get("targetRole"),
                "confidenceScore": it.get("confidenceScore", profile.get("confidenceScore", 1.0)),
                "serviceDomain": it.get("serviceDomain"),
                "subServiceDomain": it.get("subServiceDomain"),
                "topic": it.get("topic"),
                "sourceURL": it.get("sourceURL"),
                "sourceDomain": it.get("sourceDomain"),
                "sourceLicense": it.get("sourceLicense"),
                "documentType": it.get("documentType"),
                "dateCreated": it.get("dateCreated"),
                "language": it.get("language"),
            }
            rows.append(row)

        # We only use the first scraped item for this profile
        break

    return rows


def main():
    profiles = build_profiles()
    all_rows = []

    print("Generating Q&A for profiles...")
    for p in profiles:
        print("- profile:", p["targetAgeGroup"]) 
        rows = generate_for_profile(p, max_pages=1)
        all_rows.extend(rows)

    # CSV columns (explicit order)
    cols = [
        "prompt", "response",
        "targetAgeGroup", "genderIdentity", "educationBackground", "targetProfession", "digitalLiteracy",
        "geoRegion", "householdIncomeStatus", "targetRole", "confidenceScore",
        "serviceDomain", "subServiceDomain", "topic", "sourceURL", "sourceDomain", "sourceLicense",
        "documentType", "dateCreated", "language",
    ]

    print(f"Writing {len(all_rows)} rows to {OUT_CSV}")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print("Done. Output:", OUT_CSV)


if __name__ == "__main__":
    main()
