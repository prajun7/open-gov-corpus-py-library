"""
Dataset creation and management
"""

import os
import re
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .models import DatasetConfig, PromptResponse, ScrapedContent
from .scraper import scrape_website
from .exceptions import DatasetError
from .utils import ensure_directory, chunk_text

# Optional HF imports (only required when using model-based generation)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None


# -----------------------------
# Short prompt with mandatory few-shot examples
# -----------------------------
FINAL_INSTRUCTION_PROMPT = """You are an AI assistant simulating an ordinary UK resident using government services.

DO
- Generate 4 independent Q&A pairs from the INPUT TEXT.
- Questions must be realistic, practical, scenario-based (like the examples below).
- Questions must ask about definite information only (no “may/might/possibly/could”).
- Answers must use ONLY the INPUT TEXT (no outside knowledge, no assumptions).
- Use simple, clear language. Avoid jargon.
- Avoid explicit, discriminatory, sensitive, or personal data.
- Add metadata fields listed in OUTPUT KEYS.

NOTE
- Output JSON is preferred (for parsing), but exact formatting is not mandatory as it will be saved to CSV. Still: return a machine-readable JSON array if possible.

FEW-SHOT EXAMPLES (style + intent categories)

Example A (procedural / repayment)
Q: My partner and I haven’t paid council tax for a long time and I’m trying to sort it out now. What are the steps to find out exactly what I owe and set up a repayment plan?
A: Contact your local council to request a full breakdown of arrears and ask to set up an affordable repayment plan based on your income and essential costs.

Example B (transactional / application)
Q: My child needs extra support in school and I want to start the formal process. How do I request an Education, Health and Care needs assessment, and what information should I keep as evidence?
A: You can request an assessment from your local authority and keep a clear record of concerns, reports, and communications to support your request.

Example C (legal interpretation / data protection)
Q: My employer shared my salary details by mistake with other staff. Under UK rules, what kind of issue is this and who should I raise it with first?
A: This is likely a personal data breach. You should raise it with HR or the organisation’s data protection contact and follow their breach process.

Example D (informational / how it works)
Q: I use Tax-Free Childcare and want to track my payments and the government top-up. Where do I see the contribution and top-up amounts and how do I keep proof of what was paid?
A: The childcare account transaction history shows deposits and government top-ups. Save statements or screenshots regularly as proof.

TASK (repeat for 4 items)
For each of 4 iterations:
1) Create a persona by choosing:
   targetAgeGroup [under18,18-25,26-45,46-65,65+], genderIdentity [female,male,non-binary,unspecified],
   educationBackground, targetProfession, digitalLiteracy [low,medium,high],
   geoRegion [England,Scotland,Wales,Northern Ireland,other], householdIncomeStatus [under poverty limit,moderate,above moderate],
   targetRole.
2) Write:
   - prompt: a realistic citizen question (independent of other questions)
   - response: concise answer using ONLY INPUT TEXT
3) Tag:
   promptIntentType [informational,navigational,transactional,procedural,comparative,legal interpretation,personalized guidance,grievance / appeals]
   reasoningComplexity [Factual Lookup,Procedural Explanation,Multi-step Reasoning,Legal/Policy Reasoning]
   geographicContext [UK-wide,England,Scotland,Wales,Northern Ireland,N/A]
   sensitiveInformationPresent [true/false]
   vulnerableGroupTargeted [true/false]
   confidenceScore [0.0–1.0]
4) Source/provenance:
   serviceDomain (copy from INPUT TEXT), subServiceDomain (copy from INPUT TEXT), topic (copy from INPUT TEXT),
   sourceURL = "https://www.gov.uk/browse/{<copy from INPUT TEXT>}", sourceDomain="www.gov.uk",
   sourceLicense="Open Government Licence (OGL) v3.0", documentType="webpage",
   dateCreated=today (YYYY-MM-DD), language="en"

OUTPUT KEYS (each item should include)
prompt, response,
targetAgeGroup, genderIdentity, educationBackground, targetProfession, digitalLiteracy, geoRegion, householdIncomeStatus, targetRole,
promptIntentType, reasoningComplexity, geographicContext, sensitiveInformationPresent, vulnerableGroupTargeted, confidenceScore,
serviceDomain, subServiceDomain, topic, sourceURL, sourceDomain, sourceLicense, documentType, dateCreated, language

INPUT TEXT
{input_text}
"""


# -----------------------------
# Helpers
# -----------------------------
_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json(text: str) -> Optional[str]:
    if not text:
        return None
    m = _JSON_RE.search(text)
    return m.group(1).strip() if m else None


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _normalize_to_list(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [obj]
    return []


def _parse_provenance_from_text(text: str) -> Tuple[str, str, str, str, str]:
    """
    Attempts to parse:
      line0: "<serviceDomain>: <subServiceDomain>"
      line2: "<topic>"
    Returns (service_domain, sub_service_domain, topic, content_body, browse_suffix)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "Unknown", "Unknown", "Unknown", text, "unknown"

    first = lines[0]
    parts = first.split(":", 1)
    if len(parts) == 2:
        service_domain = parts[0].strip() or "Unknown"
        sub_service_domain = parts[1].strip() or "Unknown"
        topic = lines[2].strip() if len(lines) > 2 else "Unknown"
        content_body = "\n".join(lines[1:]).strip()
    else:
        service_domain = "Unknown"
        sub_service_domain = "Unknown"
        topic = lines[1].strip() if len(lines) > 1 else "Unknown"
        content_body = "\n".join(lines).strip()

    browse_suffix = service_domain.lower().replace(" ", "-").replace("&", "and")
    browse_suffix = re.sub(r"[^a-z0-9\\-]+", "-", browse_suffix).strip("-") or "unknown"
    return service_domain, sub_service_domain, topic, content_body, browse_suffix


def _merge_provenance(
    qa_obj: Dict[str, Any],
    service_domain: str,
    sub_service_domain: str,
    topic: str,
    source_url: str,
) -> Dict[str, Any]:
    out = dict(qa_obj)
    out.setdefault("serviceDomain", service_domain)
    out.setdefault("subServiceDomain", sub_service_domain)
    out.setdefault("topic", topic)

    out.setdefault("sourceURL", source_url)
    out.setdefault("sourceDomain", "www.gov.uk")
    out.setdefault("sourceLicense", "Open Government Licence (OGL) v3.0")
    out.setdefault("documentType", "webpage")
    out.setdefault("dateCreated", datetime.now().strftime("%Y-%m-%d"))
    out.setdefault("language", "en")
    return out


# -----------------------------
# Dataset Creator
# -----------------------------
class DatasetCreator:
    """Creates datasets from scraped content"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.output_dir = ensure_directory(f"OpenGovCorpus-{config.name}")

        # Model-based generation configuration (env-driven; backward compatible)
        self.use_llm = os.getenv("OPENGOV_USE_LLM", "1").strip().lower() in {"1", "true", "yes", "y"}
        self.model_name = os.getenv("OPENGOV_MODEL_NAME", "Qwen/Qwen1.5-72B-Chat")
        self.max_new_tokens = int(os.getenv("OPENGOV_MAX_NEW_TOKENS", "1024"))
        self.temperature = float(os.getenv("OPENGOV_TEMPERATURE", "0.7"))
        self.do_sample = os.getenv("OPENGOV_DO_SAMPLE", "1").strip().lower() in {"1", "true", "yes", "y"}
        self.sleep_seconds = float(os.getenv("OPENGOV_SLEEP_SECONDS", "0"))

        self._generator = None
        if self.use_llm:
            self._init_generator()

    def _init_generator(self) -> None:
        if pipeline is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            self.use_llm = False
            return

        if self._generator is not None:
            return

        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(torch.float16 if (torch is not None and device == "cuda") else None),
            device_map="auto" if device == "cuda" else None,
        )

        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
        )

    def create(self) -> dict:
        print(f"Starting dataset creation for: {self.config.name}")
        print(f"Scraping URL: {self.config.url}")

        scraped_content = scrape_website(self.config.url, max_pages=self.config.max_pages)
        if not scraped_content:
            raise DatasetError("No content scraped from website")

        print(f"Scraped {len(scraped_content)} pages")

        prompt_responses = self._create_prompt_responses(scraped_content)
        if not prompt_responses:
            raise DatasetError("No prompt-response pairs generated")

        print(f"Generated {len(prompt_responses)} prompt-response pairs")

        splits = self._split_data(prompt_responses)
        file_paths = self._save_splits(splits)

        print(f"Dataset created successfully in: {self.output_dir}")
        return file_paths

    def _generate_structured_qa_pairs_llm(self, input_text: str) -> List[Dict[str, Any]]:
        if not self.use_llm or self._generator is None:
            return []

        full_prompt = FINAL_INSTRUCTION_PROMPT.format(input_text=input_text)

        outputs = self._generator(
            full_prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )

        generated = outputs[0].get("generated_text", "") if outputs else ""
        json_str = _extract_json(generated)
        if not json_str:
            return []

        parsed = _safe_json_loads(json_str)
        items = _normalize_to_list(parsed)

        # Keep at most 4
        if len(items) > 4:
            items = items[:4]
        return items

    def _create_prompt_responses(self, content: List[ScrapedContent]) -> List[PromptResponse]:
        pairs: List[PromptResponse] = []

        for item in content:
            chunks = chunk_text(item.content, chunk_size=1000, overlap=100)

            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 100:
                    continue

                service_domain, sub_service_domain, topic, body, browse_suffix = _parse_provenance_from_text(chunk)

                browse_url = f"https://www.gov.uk/browse/{browse_suffix}"
                source_url = browse_url if service_domain != "Unknown" else item.url

                if self.use_llm and self._generator is not None:
                    llm_items: List[Dict[str, Any]] = []
                    try:
                        llm_items = self._generate_structured_qa_pairs_llm(chunk)
                    except Exception:
                        llm_items = []

                    # If model fails, fall back to legacy for this chunk
                    if llm_items:
                        for obj in llm_items:
                            prompt = str(obj.get("prompt", "")).strip()
                            response = str(obj.get("response", "")).strip()
                            if not prompt or not response:
                                continue

                            merged = _merge_provenance(obj, service_domain, sub_service_domain, topic, source_url)

                            # Put all fields (except prompt/response) in metadata so CSV stays flexible
                            metadata = {k: v for k, v in merged.items() if k not in {"prompt", "response"}}

                            pairs.append(
                                PromptResponse(
                                    prompt=prompt,
                                    response=response,
                                    metadata=metadata if self.config.include_metadata else None,
                                    url=item.url,
                                    timestamp=item.timestamp,
                                )
                            )

                        if self.sleep_seconds > 0:
                            time.sleep(self.sleep_seconds)

                        continue  # proceed to next chunk

                # Legacy fallback (no model or model failure)
                legacy_prompt = f"What information is available about '{item.title}'?"
                if len(chunks) > 1:
                    legacy_prompt += f" (Part {i+1}/{len(chunks)})"

                metadata = None
                if self.config.include_metadata:
                    metadata = {
                        "source_url": item.url,
                        "title": item.title,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "timestamp": item.timestamp.isoformat(),
                    }

                pairs.append(
                    PromptResponse(
                        prompt=legacy_prompt,
                        response=body if body else chunk,
                        metadata=metadata,
                        url=item.url,
                        timestamp=item.timestamp,
                    )
                )

        return pairs

    def _split_data(self, data: List[PromptResponse]) -> dict:
        random.shuffle(data)
        total = len(data)
        train_end = int(total * self.config.train_split)
        val_end = train_end + int(total * self.config.val_split)
        return {"train": data[:train_end], "valid": data[train_end:val_end], "test": data[val_end:]}

    def _save_splits(self, splits: dict) -> dict:
        file_paths: Dict[str, str] = {}

        for split_name, split_data in splits.items():
            filepath = self.output_dir / f"{split_name}.csv"
            rows = [item.to_dict() for item in split_data]
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)

            file_paths[split_name] = str(filepath)
            print(f"  {split_name}: {len(split_data)} samples -> {filepath}")

        return file_paths


def create_dataset(
    name: str,
    url: str,
    include_metadata: bool = True,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_pages: Optional[int] = None,
) -> dict:
    """
    Create a dataset from a government website.

    Model-based generation defaults:
      OPENGOV_USE_LLM=1
      OPENGOV_MODEL_NAME="Qwen/Qwen1.5-72B-Chat"
      OPENGOV_MAX_NEW_TOKENS=1024
      OPENGOV_TEMPERATURE=0.7
      OPENGOV_DO_SAMPLE=1
      OPENGOV_SLEEP_SECONDS=0
    """
    config = DatasetConfig(
        name=name,
        url=url,
        include_metadata=include_metadata,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        max_pages=max_pages,
    )

    creator = DatasetCreator(config)
    return creator.create()
