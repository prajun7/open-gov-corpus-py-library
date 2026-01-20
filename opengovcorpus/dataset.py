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


FEW-SHOT EXAMPLES (style + intent categories)

Example A (procedural)
Q: My partner and I haven’t paid council tax for a long time and I’m trying to sort it out now. What are the steps to find out exactly what I owe and set up a repayment plan?
A: Yes, you can usually make an arrangement to repay council tax and water arrears, but you need to contact the council and water company proactively. Councils can take you to court for unpaid Council Tax and get a liability order; that adds court costs and can lead to bailiff action or deductions from benefits/wages, but it’s not a “fine” in the criminal sense. Ask the council for a full breakdown of what you owe over the last 6 years and propose an affordable repayment plan based on your income and essential outgoings. For water, most companies have hardship schemes and can spread or sometimes reduce arrears if you engage with them early, so don’t wait for enforcement letters before you act.

Example B (procedural)
Q: For people using Tax-Free Childcare, how do you keep track of what you contribute and what the government tops up — and how do you prove what’s been paid if challenged?
A: Tax‑Free Childcare is run through an online childcare account: you pay money in, the government tops it up by 20% (up to the quarterly limit), and your provider is paid from that account. The account itself shows your contributions and the government top‑ups separately, so your transaction history is the main proof of what’s been paid. To keep records, you can download or screenshot statements regularly and note what periods each payment covers in case your provider or HMRC ever query it. Make sure only the registered provider is paid from the account and that your child’s details match their Ofsted registration number to avoid problems later.

Example C (legal interpretation)
Q: I’ve had a death in the family and I’m overwhelmed by all the admin — registering the death, arranging the funeral, dealing with estates and accounts. For those who’ve been through it, what order did you do things in and what helped you cope?
A: After a death in England or Wales you normally: register the death within 5 days with the local register office, use the “Tell Us Once” service to notify government departments, then arrange the funeral (through a funeral director or privately). At the same time, the executor or next of kin collects important documents (will, death certificate, bank details, property deeds) and contacts banks, pension providers, landlords or mortgage lenders and insurers to freeze or transfer accounts. If there is a will, the named executors may need to apply for probate before they can sell property or distribute the estate; if there isn’t, the intestacy rules decide who can act and who inherits. Keeping a simple checklist and asking for help from a trusted friend, solicitor or advice agency can make the process feel more manageable.

Example D (informational)
Q: I’m 15 and 9 months and can apply for my provisional licence, but the rules say I can only drive a car at 17. Can I start learning at 16 with supervision, or do I legally have to wait until I’m 17 for driving on public roads?
A: In the UK you can apply for a provisional car licence at 15 years and 9 months, but you can’t drive a car on public roads until you’re 17, even with supervision, unless you qualify under specific disability‑related rules. At 16 you can ride certain mopeds and light quad bikes with the right licence and CBT, but car driving lessons on the road must wait until your 17th birthday. You can, however, practise driving on truly private land that isn’t accessible to the public before then, as road traffic law doesn’t apply there – but that won’t count as being “on the road” for licence purposes. Once you’re 17, you can drive a car with L‑plates while supervised by someone who meets the DVLA rules and is properly insured.

TASK (repeat for 4 items)
For each of 4 iterations:
1) Create a persona by choosing:
    targetAgeGroup [under18,18-25,26-45,46-65,65+], genderIdentity [female,male,non-binary,unspecified],
    educationBackground, targetProfession, digitalLiteracy [low,medium,high],
    geoRegion [England,Scotland,Wales,Northern Ireland,other], householdIncomeStatus [under poverty limit,moderate,above moderate],
    targetRole.
2) Write:
    - prompt: a realistic citizen question that your chosen persona would ask (independent of other questions)
    - response: concise answer tailored to your chosen persona using ONLY INPUT TEXT
3) Tag:
    promptIntentType [informational,procedural,comparative]
    geographicContext [UK-wide,England,Scotland,Wales,Northern Ireland]
    sensitiveInformationPresent [true/false]
    vulnerableGroupTargeted [true/false]
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
{persona_block}
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


def define_config(
    targetAgeGroup: str,
    genderIdentity: str,
    educationBackground: str,
    targetProfession: str,
    digitalLiteracy: str,
    geoRegion: str,
    householdIncomeStatus: str,
    targetRole: str,
    promptIntentType: str = "informational",
    geographicContext: str = "UK-wide",
    sensitiveInformationPresent: bool = False,
    vulnerableGroupTargeted: bool = False,
    confidenceScore: float = 1.0,
) -> Dict[str, Any]:
    """Create a persona configuration dictionary used to customise the LLM prompt.

    This helper returns a JSON-serializable dict containing the persona fields and
    simple tags. The resulting dict can be passed to ``create_dataset(..., persona_config=...)``
    or used directly when formatting the instruction prompt.
    """
    return {
        "targetAgeGroup": targetAgeGroup,
        "genderIdentity": genderIdentity,
        "educationBackground": educationBackground,
        "targetProfession": targetProfession,
        "digitalLiteracy": digitalLiteracy,
        "geoRegion": geoRegion,
        "householdIncomeStatus": householdIncomeStatus,
        "targetRole": targetRole,
        "promptIntentType": promptIntentType,
        "geographicContext": geographicContext,
        "sensitiveInformationPresent": sensitiveInformationPresent,
        "vulnerableGroupTargeted": vulnerableGroupTargeted,
        "confidenceScore": confidenceScore,
    }


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

        # Some HF-hosted instruction/chat models include custom code in the repo
        # (e.g. special tokenizers or model wrappers). Hugging Face requires
        # explicit approval to run that code for security reasons.
        # Only set trust_remote_code=True for checkpoints you trust.
        # Try the default (may use a fast Rust tokenizer). If that fails because
        # the remote repo defines a Python-only tokenizer class or the fast
        # tokenizer isn't available, fall back to use_fast=False.
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, use_fast=False
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(torch.float16 if (torch is not None and device == "cuda") else None),
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        # Create the text-generation pipeline. If the model was loaded with
        # `accelerate` (device_map set) the pipeline must NOT be given a
        # explicit `device` argument — that raises a ValueError. Detect the
        # presence of a device map on the model and avoid passing `device` in
        # that case. As an extra safeguard, attempt with `device` first and
        # fall back to creating the pipeline without it if a ValueError occurs.
        device_arg = 0 if device == "cuda" else -1
        try:
            if getattr(model, "hf_device_map", None):
                # model already placed with accelerate/device_map; omit device
                self._generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
            else:
                # safe to pass a device index
                self._generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=device_arg,
                )
        except ValueError:
            # Some environments may still raise if device disagrees with model;
            # retry without device argument so pipeline uses model's placement.
            self._generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
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

        # Build persona block if provided in DatasetConfig
        persona_cfg = getattr(self.config, "persona_config", None)
        persona_block = ""
        if persona_cfg:
            try:
                # Present persona config as a short labelled JSON block the model can follow
                persona_block = "\nPERSONA CONFIGURATION:\n" + json.dumps(persona_cfg, indent=2) + "\n"
            except Exception:
                persona_block = ""

        full_prompt = FINAL_INSTRUCTION_PROMPT.format(input_text=input_text, persona_block=persona_block)

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
    persona_config: Optional[Dict[str, Any]] = None,
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
        persona_config=persona_config,
    )

    creator = DatasetCreator(config)
    return creator.create()
