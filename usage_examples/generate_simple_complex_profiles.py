# -*- coding: utf-8 -*-
"""
Deterministically crawl gov.uk benefits browse pages and generate ~2000 Q&A rows.

Requirements (per user):
1) Scrape at least ALL pages under: https://www.gov.uk/browse/benefits
2) Generate 10–12 Q&A items per page (not random pages)
3) Total prompts in CSV ~2000

Behavior:
- Crawls ALL URLs under /browse/benefits (including nested browse pages).
- Optionally expands to additional gov.uk pages linked from those browse pages (e.g. /benefits/...)
  only until the dataset reaches ~2000 prompts, so totals stay bounded.
- For each crawled page, generates exactly N Q&A items where N is chosen as 10 or 12
  to keep the overall total near TARGET_TOTAL_ROWS, while honoring the 10–12 per page constraint.
- Uses opengovcorpus DatasetCreator LLM generator if available; otherwise uses a deterministic heuristic.
- Writes CSV incrementally.

Run:
  python3 usage_examples/generate_benefits_qas_2000.py
"""

import csv
import json
import re
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime
from itertools import product, cycle
from collections import deque
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

# Ensure the project root is on sys.path so the local opengovcorpus package is imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opengovcorpus as og
from opengovcorpus.models import DatasetConfig
from opengovcorpus.dataset import DatasetCreator, _parse_provenance_from_text


# -----------------------------
# Configuration
# -----------------------------
BASE = "https://www.gov.uk"
START_BROWSE = "https://www.gov.uk/browse/benefits"

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "gov_uk_benefits_qas.csv"

# Target total rows ~2000
TARGET_TOTAL_ROWS = 2000

# Crawl controls
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS_SEC = 0.2
MAX_CRAWL_PAGES_HARD_CAP = 5000  # safety cap; should not be hit in practice
MAX_CRAWL_DEPTH = 6              # enough to traverse browse tree; keep bounded

# Page content controls
MIN_TEXT_CHARS = 400  # skip pages with too-little text
LANGUAGE = "en"

# Persona configuration grid (kept moderate).
# This script cycles profiles across pages; it guarantees each profile appears at least once
# if there are enough pages.
AGE_GROUPS = ["under18", "18-25", "26-45", "46-65", "65+"]

CONFIG_SPACE = {
    "genderIdentity": ["male", "female"],
    "educationBackground": ["secondary", "tertiary"],
    "digitalLiteracy": ["low", "medium", "high"],
    "geoRegion": ["England", "Scotland", "Wales"],
    "householdIncomeStatus": ["moderate", "above moderate"],
    "targetRole": ["service-user"],
    "targetProfession": ["student", "retail", "nurse", "teacher", "retired"],
    "promptIntentType": ["procedural", "instructional"],  # requested
}


# -----------------------------
# Helpers
# -----------------------------
def stable_id(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:10]


def is_same_domain(url: str) -> bool:
    try:
        return urlparse(url).netloc == urlparse(BASE).netloc
    except Exception:
        return False


def normalize_url(url: str) -> str:
    # remove fragment; drop trailing slash normalization is handled by gov.uk canonicalization
    url, _frag = urldefrag(url)
    return url


def is_html_like_url(url: str) -> bool:
    # exclude common non-html
    lowered = url.lower()
    if any(lowered.endswith(ext) for ext in [".pdf", ".csv", ".zip", ".jpg", ".jpeg", ".png", ".gif", ".svg"]):
        return False
    return True


def extract_links(html: str, current_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        abs_url = urljoin(current_url, href)
        abs_url = normalize_url(abs_url)
        if not abs_url.startswith(BASE):
            continue
        if not is_html_like_url(abs_url):
            continue
        links.append(abs_url)
    return links


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Prefer <main>, otherwise fall back to #content or body
    main = soup.find("main")
    if main is None:
        main = soup.find(id="content")
    if main is None:
        main = soup.body if soup.body is not None else soup

    # Remove scripts/styles/nav/footer forms
    for tag in main.select("script, style, nav, footer, header, form, aside"):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch(url: str, session: requests.Session) -> tuple[str, str]:
    r = session.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text, r.url  # return final URL after redirects


# -----------------------------
# Crawl strategy
# -----------------------------
def crawl_benefits_pages() -> tuple[list[str], list[str]]:
    """
    Returns:
      browse_urls: all URLs under /browse/benefits (must include all)
      extra_urls: additional gov.uk URLs linked from browse pages (e.g., /benefits/...) excluding browse_urls
    """
    session = requests.Session()

    visited = set()
    browse_urls = []
    extra_candidates = set()

    q = deque()
    q.append((START_BROWSE, 0))

    while q and len(visited) < MAX_CRAWL_PAGES_HARD_CAP:
        url, depth = q.popleft()
        url = normalize_url(url)
        if url in visited:
            continue
        visited.add(url)

        # Polite pacing
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

        try:
            html, final_url = fetch(url, session)
        except Exception:
            continue

        final_url = normalize_url(final_url)

        # Track browse subtree membership based on final URL
        if final_url.startswith(f"{BASE}/browse/benefits"):
            browse_urls.append(final_url)

        # Stop traversing too deep
        if depth >= MAX_CRAWL_DEPTH:
            # Still collect outbound links as extras, but don't enqueue
            links = extract_links(html, final_url)
            for lk in links:
                if is_same_domain(lk):
                    if lk.startswith(f"{BASE}/browse/benefits"):
                        # If we didn't enqueue due to depth, we still want to ensure coverage.
                        # Enqueue browse pages even beyond depth, but bounded by hard cap.
                        if lk not in visited:
                            q.append((lk, depth + 1))
                    else:
                        extra_candidates.add(lk)
            continue

        links = extract_links(html, final_url)

        # Deterministic ordering for reproducibility
        for lk in sorted(set(links)):
            if not is_same_domain(lk):
                continue
            # Always enqueue browse subtree pages to ensure "all pages under /browse/benefits"
            if lk.startswith(f"{BASE}/browse/benefits"):
                if lk not in visited:
                    q.append((lk, depth + 1))
            else:
                # Collect other pages linked from browse pages as extras (e.g., /benefits/...)
                # Only if we are currently in browse subtree
                if final_url.startswith(f"{BASE}/browse/benefits"):
                    extra_candidates.add(lk)

    # Deduplicate + deterministic order
    browse_urls = sorted(set(browse_urls))
    extra_urls = sorted(extra_candidates - set(browse_urls))

    return browse_urls, extra_urls


# -----------------------------
# Persona profiles
# -----------------------------
def build_profiles(age_groups=None, config_space=None) -> list[dict]:
    age_groups = age_groups or AGE_GROUPS
    config_space = config_space or CONFIG_SPACE

    keys = list(config_space.keys())
    combos = list(product(*[config_space[k] for k in keys]))

    profiles = []
    for age in age_groups:
        for combo in combos:
            kwargs = dict(zip(keys, combo))
            p = og.define_config(targetAgeGroup=age, **kwargs)
            p["profileId"] = stable_id(p)
            profiles.append(p)
    return profiles


# -----------------------------
# Q&A generation
# -----------------------------
def llm_generate_items(creator: DatasetCreator, instruction: str) -> list[dict]:
    gen = getattr(creator, "_generator", None)
    if not getattr(creator, "use_llm", False) or gen is None:
        return []

    try:
        outputs = gen(
            instruction,
            max_new_tokens=getattr(creator, "max_new_tokens", 1024),
            do_sample=getattr(creator, "do_sample", True),
            temperature=getattr(creator, "temperature", 0.7),
        )
        text = outputs[0].get("generated_text", "") if outputs else ""
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return parsed
    except Exception:
        return []
    return []


def heuristic_generate_items(input_text: str, page_url: str, profile: dict, n_items: int) -> list[dict]:
    """
    Deterministic heuristic: derive Q&A from headings/bullets/paragraphs.
    Produces exactly n_items items.
    """
    now = datetime.now().strftime("%Y-%m-%d")

    # Provenance inference (best-effort)
    serviceDomain, subServiceDomain, topic, _, browse_suffix = _parse_provenance_from_text(input_text)
    sourceURL = f"{BASE}/browse/{browse_suffix}" if serviceDomain != "Unknown" and browse_suffix else page_url

    # Split into candidate "facts" from paragraphs/lines
    lines = [ln.strip() for ln in input_text.splitlines() if ln.strip()]
    # Keep only reasonably informative lines
    facts = [ln for ln in lines if len(ln) >= 40][:200]

    if not facts:
        facts = [input_text[:400]]

    items = []
    # Use stride to pick facts deterministically across the page
    stride = max(1, len(facts) // n_items)
    picked = [facts[i] for i in range(0, min(len(facts), stride * n_items), stride)]
    # Pad if needed
    while len(picked) < n_items:
        picked.append(facts[min(len(facts) - 1, 0)])

    for i in range(n_items):
        fact = picked[i]
        # Alternate simple/complex deterministically
        if i % 2 == 0:
            prompt = f"What does this page say about: '{fact[:90]}'?"
            response = fact
        else:
            # multi-step style: refer to two nearby facts if possible
            j = min(i, len(picked) - 1)
            k = min(j + 1, len(picked) - 1)
            context = (picked[j] + " " + picked[k]).strip()
            prompt = f"Based on this page, how should I proceed if my situation matches: '{context[:120]}'?"
            response = context

        items.append(
            {
                "prompt": prompt,
                "response": response,
                "confidenceScore": float(profile.get("confidenceScore", 1.0)),
                "serviceDomain": serviceDomain,
                "subServiceDomain": subServiceDomain,
                "topic": topic,
                "sourceURL": sourceURL,
                "sourceDomain": "www.gov.uk",
                "sourceLicense": "Open Government Licence (OGL) v3.0",
                "documentType": "webpage",
                "dateCreated": now,
                "language": LANGUAGE,
            }
        )
    return items


def generate_qas_for_page(page_url: str, page_text: str, profile: dict, n_items: int) -> list[dict]:
    """
    Generates exactly n_items Q&A dicts for a page + persona profile.
    """
    cfg = DatasetConfig(
        name=f"tmp-benefits-{profile.get('profileId','')}",
        url=page_url,
        max_pages=1,
        persona_config=profile,
    )
    creator = DatasetCreator(cfg)

    persona_json = json.dumps(profile, indent=2)
    instruction = (
    "You are an AI assistant simulating an ordinary UK resident using government services.\n\n"
    "DO\n"
    f"- Generate exactly {n_items} Q&A items from the INPUT TEXT.\n"
    "- Questions must be realistic, practical, scenario-based (like the examples below).\n"
    "- Questions must ask about definite information only (no “may/might/possibly/could”).\n"
    "- Answers must use ONLY the INPUT TEXT (no outside knowledge, no assumptions).\n"
    "- Use simple, clear language. Avoid jargon.\n"
    "- Avoid explicit, discriminatory, sensitive, or personal data.\n\n"
    "FEW-SHOT EXAMPLES (style + intent categories)\n\n"
    "Example A (procedural)\n"
    "Q: My partner and I haven’t paid council tax for a long time and I’m trying to sort it out now. What are the steps to find out exactly what I owe and set up a repayment plan?\n"
    "A: Yes, you can usually make an arrangement to repay council tax and water arrears, but you need to contact the council and water company proactively. Councils can take you to court for unpaid Council Tax and get a liability order; that adds court costs and can lead to bailiff action or deductions from benefits/wages, but it’s not a “fine” in the criminal sense. Ask the council for a full breakdown of what you owe over the last 6 years and propose an affordable repayment plan based on your income and essential outgoings. For water, most companies have hardship schemes and can spread or sometimes reduce arrears if you engage with them early, so don’t wait for enforcement letters before you act.\n\n"
    "Example B (procedural)\n"
    "Q: For people using Tax-Free Childcare, how do you keep track of what you contribute and what the government tops up — and how do you prove what’s been paid if challenged?\n"
    "A: Tax-Free Childcare is run through an online childcare account: you pay money in, the government tops it up by 20% (up to the quarterly limit), and your provider is paid from that account. The account itself shows your contributions and the government top-ups separately, so your transaction history is the main proof of what’s been paid. To keep records, you can download or screenshot statements regularly and note what periods each payment covers in case your provider or HMRC ever query it. Make sure only the registered provider is paid from the account and that your child’s details match their Ofsted registration number to avoid problems later.\n\n"
    "Example C (legal interpretation)\n"
    "Q: I’ve had a death in the family and I’m overwhelmed by all the admin — registering the death, arranging the funeral, dealing with estates and accounts. For those who’ve been through it, what order did you do things in and what helped you cope?\n"
    "A: After a death in England or Wales you normally: register the death within 5 days with the local register office, use the “Tell Us Once” service to notify government departments, then arrange the funeral (through a funeral director or privately). At the same time, the executor or next of kin collects important documents (will, death certificate, bank details, property deeds) and contacts banks, pension providers, landlords or mortgage lenders and insurers to freeze or transfer accounts. If there is a will, the named executors may need to apply for probate before they can sell property or distribute the estate; if there isn’t, the intestacy rules decide who can act and who inherits. Keeping a simple checklist and asking for help from a trusted friend, solicitor or advice agency can make the process feel more manageable.\n\n"
    "Example D (informational)\n"
    "Q: I’m 15 and 9 months and can apply for my provisional licence, but the rules say I can only drive a car at 17. Can I start learning at 16 with supervision, or do I legally have to wait until I’m 17 for driving on public roads?\n"
    "A: In the UK you can apply for a provisional car licence at 15 years and 9 months, but you can’t drive a car on public roads until you’re 17, even with supervision, unless you qualify under specific disability-related rules. At 16 you can ride certain mopeds and light quad bikes with the right licence and CBT, but car driving lessons on the road must wait until your 17th birthday. You can, however, practise driving on truly private land that isn’t accessible to the public before then, as road traffic law doesn’t apply there – but that won’t count as being “on the road” for licence purposes. Once you’re 17, you can drive a car with L-plates while supervised by someone who meets the DVLA rules and is properly insured.\n\n"
    "TASK\n"
    f"- Create {n_items} independent Q&A items.\n"
    "- Each question should be written from the given PERSONA.\n"
    "- Tag each item with: promptIntentType [informational,procedural,comparative], geographicContext [UK-wide,England,Scotland,Wales,Northern Ireland], sensitiveInformationPresent [true/false], vulnerableGroupTargeted [true/false].\n\n"
    "OUTPUT FORMAT\n"
    "Output a JSON array. For each item include these keys:\n"
    "  prompt, response,\n"
    "  targetAgeGroup, genderIdentity, educationBackground, targetProfession, digitalLiteracy, geoRegion, householdIncomeStatus, targetRole,\n"
    "  promptIntentType, reasoningComplexity, geographicContext, sensitiveInformationPresent, vulnerableGroupTargeted, confidenceScore,\n"
    "  serviceDomain, subServiceDomain, topic, sourceURL, sourceDomain, sourceLicense, documentType, dateCreated, language\n\n"
    "PERSONA:\n" + persona_json + "\n\n"
    "SOURCE URL:\n" + page_url + "\n\n"
    "INPUT TEXT:\n" + page_text
)

    items = llm_generate_items(creator, instruction)
    # Validate length; if wrong or empty, fall back
    if not items or not isinstance(items, list) or len(items) != n_items:
        items = heuristic_generate_items(page_text, page_url, profile, n_items)

    # Normalize minimally
    normed = []
    for it in items[:n_items]:
        normed.append(
            {
                "prompt": it.get("prompt", ""),
                "response": it.get("response", ""),
                "confidenceScore": it.get("confidenceScore", profile.get("confidenceScore", 1.0)),
                "serviceDomain": it.get("serviceDomain", ""),
                "subServiceDomain": it.get("subServiceDomain", ""),
                "topic": it.get("topic", ""),
                "sourceURL": it.get("sourceURL", page_url),
                "sourceDomain": it.get("sourceDomain", "www.gov.uk"),
                "sourceLicense": it.get("sourceLicense", "Open Government Licence (OGL) v3.0"),
                "documentType": it.get("documentType", "webpage"),
                "dateCreated": it.get("dateCreated", datetime.now().strftime("%Y-%m-%d")),
                "language": it.get("language", LANGUAGE),
            }
        )
    # Ensure exact n
    if len(normed) != n_items:
        normed = (normed + heuristic_generate_items(page_text, page_url, profile, n_items))[:n_items]
    return normed


# -----------------------------
# Main
# -----------------------------
def choose_items_per_page(num_pages_planned: int) -> int:
    """
    Choose 10 or 12 per page so total is near TARGET_TOTAL_ROWS.
    Must return within [10, 12].
    """
    if num_pages_planned <= 0:
        return 10
    # Ideal items/page
    ideal = TARGET_TOTAL_ROWS / num_pages_planned
    # Clamp to 10–12
    if ideal <= 10:
        return 10
    if ideal >= 12:
        return 12
    # Round to nearest of {10, 12} with preference to closer
    return 10 if abs(ideal - 10) <= abs(ideal - 12) else 12


def main():
    print("Crawling benefits browse pages...")
    browse_urls, extra_urls = crawl_benefits_pages()
    print(f"Found {len(browse_urls)} browse URLs under /browse/benefits")
    print(f"Found {len(extra_urls)} extra candidate URLs linked from browse pages")

    profiles = build_profiles()
    print(f"Built {len(profiles)} persona profiles from config grid")

    # Plan page list:
    # 1) All browse urls (must include all)
    # 2) Optionally add extra urls until we approach TARGET_TOTAL_ROWS
    planned_urls = list(browse_urls)

    # Decide per-page QAs based on browse-only first
    n_per_page = choose_items_per_page(len(planned_urls))

    # If browse-only total is below target, extend with extra URLs
    est_total = len(planned_urls) * n_per_page
    if est_total < TARGET_TOTAL_ROWS and extra_urls:
        # How many extra pages can we add while respecting 10–12 constraint?
        # Keep n_per_page fixed (10 or 12), add pages until est_total near target
        remaining = TARGET_TOTAL_ROWS - est_total
        add_pages = max(0, remaining // n_per_page)
        planned_urls.extend(extra_urls[:add_pages])

    # Recompute n_per_page once after extending (still 10–12)
    n_per_page = choose_items_per_page(len(planned_urls))

    # Final planned estimate
    est_total = len(planned_urls) * n_per_page
    print(f"Planning to process {len(planned_urls)} pages with {n_per_page} QAs/page -> ~{est_total} rows")

    cols = [
        "prompt", "response",
        "profileId",
        "targetAgeGroup", "genderIdentity", "educationBackground", "targetProfession", "digitalLiteracy",
        "geoRegion", "householdIncomeStatus", "targetRole", "confidenceScore",
        "serviceDomain", "subServiceDomain", "topic",
        "pageURL", "sourceURL", "sourceDomain", "sourceLicense",
        "documentType", "dateCreated", "language",
    ]

    session = requests.Session()

    # Persona assignment across pages:
    # - Try to cover each profile at least once if possible, otherwise cycle.
    profile_iter = cycle(profiles)

    total_written = 0
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for idx, page_url in enumerate(planned_urls, 1):
            # Stop if we already hit target (keep "around 2000")
            if total_written >= TARGET_TOTAL_ROWS:
                break

            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
            try:
                html, final_url = fetch(page_url, session)
                text = extract_main_text(html)
                if len(text) < MIN_TEXT_CHARS:
                    continue
            except Exception:
                continue

            profile = next(profile_iter)

            qas = generate_qas_for_page(final_url, text, profile, n_items=n_per_page)

            # Write rows
            for qa in qas:
                if total_written >= TARGET_TOTAL_ROWS:
                    break

                row = {
                    "prompt": qa["prompt"],
                    "response": qa["response"],

                    "profileId": profile.get("profileId"),
                    "targetAgeGroup": profile.get("targetAgeGroup"),
                    "genderIdentity": profile.get("genderIdentity"),
                    "educationBackground": profile.get("educationBackground"),
                    "targetProfession": profile.get("targetProfession"),
                    "digitalLiteracy": profile.get("digitalLiteracy"),
                    "geoRegion": profile.get("geoRegion"),
                    "householdIncomeStatus": profile.get("householdIncomeStatus"),
                    "targetRole": profile.get("targetRole"),

                    "confidenceScore": qa.get("confidenceScore", profile.get("confidenceScore", 1.0)),

                    "serviceDomain": qa.get("serviceDomain"),
                    "subServiceDomain": qa.get("subServiceDomain"),
                    "topic": qa.get("topic"),

                    "pageURL": final_url,
                    "sourceURL": qa.get("sourceURL", final_url),
                    "sourceDomain": qa.get("sourceDomain", "www.gov.uk"),
                    "sourceLicense": qa.get("sourceLicense", "Open Government Licence (OGL) v3.0"),
                    "documentType": qa.get("documentType", "webpage"),
                    "dateCreated": qa.get("dateCreated"),
                    "language": qa.get("language", LANGUAGE),
                }
                writer.writerow(row)
                total_written += 1

            if idx % 25 == 0:
                print(f"Processed {idx}/{len(planned_urls)} pages; wrote {total_written} rows")

    print(f"Done. Wrote {total_written} rows to: {OUT_CSV}")
    if total_written < TARGET_TOTAL_ROWS:
        print(
            f"Note: total rows < {TARGET_TOTAL_ROWS}. "
            f"Either the crawl yielded fewer usable pages or many pages had low text content."
        )


if __name__ == "__main__":
    main()