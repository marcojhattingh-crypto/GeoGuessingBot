#!/usr/bin/env python3
# Build geo_tables.json for StreetCLIP
# - Scrapes Street View coverage countries from Wikipedia (with fallback)
# - Fetches ADM1 regions and Top-N largest cities by population from Wikidata
# - Cleans ALL labels (continents, countries, regions, cities) aggressively
#
# Usage:
#   python scripts/build_geo_tables.py --top-cities 5
# Optional:
#   --include-towns  # include towns (Q3957) in the top-N ranking set
#   --out geo_tables.json
#
# Requires: requests, tqdm

from __future__ import annotations
import argparse, json, re, time, html
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

WIKI_COVERAGE_URL = "https://en.wikipedia.org/wiki/Google_Street_View_coverage"
USER_AGENT = "StreetCLIP-Builder/2.0 (contact: none)"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# ---------------------- Minimal fallback (if Wikipedia parse fails) ----------------------
FALLBACK_COVERED = OrderedDict({
    "Africa": ["South Africa","Namibia","Ghana","Tunisia","Uganda","Nigeria","Morocco","Kenya","Egypt"],
    "Asia": ["Japan","South Korea","Taiwan","Hong Kong","Macau","Thailand","Malaysia","Indonesia",
             "Philippines","Singapore","Vietnam","Cambodia","Laos","India","Sri Lanka","Bangladesh",
             "Nepal","Bhutan","Mongolia","Kazakhstan","Kyrgyzstan","Turkey","United Arab Emirates",
             "Qatar","Jordan","Israel","Palestine","Lebanon","Kuwait","Oman"],
    "Europe": ["Albania","Andorra","Austria","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria",
               "Croatia","Czechia","Denmark","Estonia","Faroe Islands","Finland","France","Germany",
               "Gibraltar","Greece","Hungary","Iceland","Ireland","Italy","Kosovo","Latvia",
               "Liechtenstein","Lithuania","Luxembourg","Malta","Monaco","Montenegro","Netherlands",
               "North Macedonia","Norway","Poland","Portugal","Romania","Serbia","Slovakia","Slovenia",
               "Spain","Svalbard and Jan Mayen","Sweden","Switzerland","Ukraine","United Kingdom","Åland Islands"],
    "North America": ["United States","Canada","Mexico","Guatemala","Costa Rica","Dominican Republic",
                      "Bermuda","Puerto Rico","United States Virgin Islands","Greenland","St. Pierre and Miquelon","Martinique"],
    "South America": ["Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","Peru","Paraguay","Uruguay"],
    "Oceania": ["Australia","New Zealand","American Samoa","Northern Mariana Islands","Guam","Samoa","Vanuatu"]
})

# ---------------------- Sanitizers ----------------------
_TAG_RE = re.compile(r"<[^>]*>")
_BRACKET_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]")
_MULTI_SPACE_RE = re.compile(r"\s+")

def _multi_unescape(s: str, passes: int = 3) -> str:
    prev = s
    for _ in range(passes):
        s2 = html.unescape(prev)
        if s2 == prev:
            break
        prev = s2
    return prev

def sanitize_label_soft(s: str) -> str:
    """Soft cleaner used during build (keeps punctuation that might help lookups)."""
    if not s:
        return s
    s = _multi_unescape(s)
    s = _TAG_RE.sub("", s)
    s = s.replace("\u200b", "")
    s = s.strip()
    s = re.sub(r"^\*+\s*", "", s)       # leading stars
    s = re.sub(r"\s*\*+$", "", s)       # trailing stars
    s = _BRACKET_FOOTNOTE_RE.sub("", s) # [123]
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s

def brutal_clean(s: str) -> str:
    """Final-pass cleaner: keep letters (incl. accents), spaces, hyphens, apostrophes. Drop everything else."""
    if not s:
        return s
    s = _multi_unescape(s)
    s = _TAG_RE.sub("", s)
    s = s.replace("\u200b", "")
    # remove any HTML entities like &#91; or &amp;
    s = re.sub(r"&[#A-Za-z0-9]+;", "", s)
    # allow only letters (ASCII + Latin-1 accents), space, hyphen, apostrophe
    s = re.sub(r"[^A-Za-zÀ-ÿ' -]+", "", s)
    # collapse whitespace and strip stray leading/trailing apostrophes/hyphens
    s = _MULTI_SPACE_RE.sub(" ", s).strip(" '-").strip()
    return s

# ---------------------- HTTP & SPARQL ----------------------
def http_get(url, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["User-Agent"] = USER_AGENT
    return requests.get(url, headers=headers, timeout=30, **kwargs)

def sparql(query: str):
    r = http_get(SPARQL_ENDPOINT, params={"query": query, "format": "json"})
    r.raise_for_status()
    return r.json()["results"]["bindings"]

# ---------------------- Scrape Street View coverage (Wikipedia) ----------------------
def parse_coverage_from_wikipedia() -> OrderedDict[str, List[str]]:
    """Scrape coverage table → {continent: [countries...]}; fallback if parse too small."""
    try:
        html_text = http_get(WIKI_COVERAGE_URL).text
        rows = re.findall(r"<tr>.*?</tr>", html_text, flags=re.S | re.I)
        raw_pairs = []
        for tr in rows:
            cols = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S | re.I)
            cols = [sanitize_label_soft(c) for c in cols]
            if len(cols) >= 2:
                country, cont = cols[0], cols[1]
                if country and cont and cont not in ["", "—", "-"]:
                    raw_pairs.append((country, cont))

        # Normalize continent naming
        def norm_cont(c: str) -> str | None:
            c = sanitize_label_soft(c)
            mapping = {
                "Europe": "Europe",
                "Asia": "Asia",
                "Africa": "Africa",
                "North America": "North America",
                "America Central": "North America",
                "Central America": "North America",
                "South America": "South America",
                "Oceania": "Oceania",
                "Australasia": "Oceania",
            }
            return mapping.get(c, None)

        by_cont = OrderedDict({k: [] for k in FALLBACK_COVERED.keys()})
        seen = {k: set() for k in by_cont.keys()}
        for country, cont in raw_pairs:
            cont_norm = norm_cont(cont)
            if cont_norm in by_cont and country:
                country = sanitize_label_soft(country)
                if country and country not in seen[cont_norm]:
                    by_cont[cont_norm].append(country)
                    seen[cont_norm].add(country)

        # Basic size check
        total = sum(len(v) for v in by_cont.values())
        if total < 40:
            return FALLBACK_COVERED

        # Soft-clean pass done; return
        return by_cont
    except Exception:
        return FALLBACK_COVERED

# ---------------------- Wikidata lookups ----------------------
def wikidata_country_qid(name: str) -> str | None:
    # Try exact label; fallback to alias
    q = f'''
    SELECT ?item WHERE {{
      {{
        ?item rdfs:label "{name}"@en.
      }} UNION {{
        ?item skos:altLabel "{name}"@en.
      }}
      ?item wdt:P31/wdt:P279* wd:Q6256.
    }} LIMIT 1
    '''
    res = sparql(q)
    if not res:
        return None
    return res[0]["item"]["value"].split("/")[-1]

def wikidata_adm1_regions(country_qid: str) -> List[Tuple[str, float | None, float | None]]:
    q = f"""
    SELECT ?r ?rLabel ?lat ?lon WHERE {{
      ?r (wdt:P31/wdt:P279*) wd:Q10864048 .
      ?r wdt:P17 wd:{country_qid} .
      OPTIONAL {{ ?r wdt:P625 ?coord .
                 BIND(geof:latitude(?coord) AS ?lat)
                 BIND(geof:longitude(?coord) AS ?lon) }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    out = []
    for b in sparql(q):
        name = sanitize_label_soft(b["rLabel"]["value"])
        lat = float(b["lat"]["value"]) if "lat" in b else None
        lon = float(b["lon"]["value"]) if "lon" in b else None
        if name:
            out.append((name, lat, lon))
    return out

def wikidata_top_cities(country_qid: str, top_n: int = 5, include_towns: bool = False) -> List[Tuple[str, int, float, float]]:
    # Candidate types
    type_clause = """
      ?pl wdt:P31/wdt:P279* wd:Q515 .   # city
    """
    if include_towns:
        type_clause = """
          { ?pl wdt:P31/wdt:P279* wd:Q515 . } UNION { ?pl wdt:P31/wdt:P279* wd:Q3957 . }  # city or town
        """

    q = f"""
    SELECT ?name ?pop ?lat ?lon WHERE {{
      ?pl wdt:P17 wd:{country_qid} .
      {type_clause}
      ?pl wdt:P1082 ?pop .
      ?pl wdt:P625 ?coord .
      BIND(geof:latitude(?coord) AS ?lat)
      BIND(geof:longitude(?coord) AS ?lon)
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?pl rdfs:label ?name.
      }}
    }}
    ORDER BY DESC(?pop)
    LIMIT {int(top_n)}
    """
    out = []
    for b in sparql(q):
        name = sanitize_label_soft(b["name"]["value"])
        if not name:
            continue
        lat = float(b["lat"]["value"])
        lon = float(b["lon"]["value"])
        pop = int(float(b["pop"]["value"]))
        out.append((name, pop, lat, lon))
    return out

# ---------------------- Final cleanup utilities ----------------------
def clean_list(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for v in values:
        vv = brutal_clean(v)
        if vv and vv not in seen:
            out.append(vv)
            seen.add(vv)
    return out

def clean_dict_of_lists(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = OrderedDict()
    for k, lst in d.items():
        kk = brutal_clean(k)
        if not kk:
            continue
        cleaned[kk] = clean_list(lst)
    return cleaned

def clean_coord_dict(d: Dict[str, List[float] | Tuple[float, float]]) -> Dict[str, List[float]]:
    cleaned: Dict[str, List[float]] = OrderedDict()
    for k, v in d.items():
        kk = brutal_clean(k)
        if not kk:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                cleaned[kk] = [float(v[0]), float(v[1])]
            except Exception:
                pass
    return cleaned

# ---------------------- Main build ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-cities", type=int, default=5, help="Pick top-N largest cities per country (by population).")
    ap.add_argument("--include-towns", action="store_true", help="Include towns (Q3957) in the ranking set.")
    ap.add_argument("--out", default="geo_tables.json", help="Output JSON path.")
    args = ap.parse_args()

    # 1) Coverage
    coverage_soft = parse_coverage_from_wikipedia()  # soft-cleaned dict

    # 2) Prepare outputs (we'll fill then brutally clean globally before write)
    out_CONTINENTS: List[str] = list(coverage_soft.keys())
    out_COUNTRIES: Dict[str, List[str]] = coverage_soft

    COUNTRY_CENTROIDS: Dict[str, List[float]] = {}
    REGIONS_BY_COUNTRY: Dict[str, List[str]] = {}
    REGION_CENTROIDS: Dict[str, List[float]] = {}
    CITIES_BY_COUNTRY: Dict[str, List[str]] = {}
    CITY_COORDS: Dict[str, List[float]] = {}

    # 3) Per-country data
    for cont, countries in coverage_soft.items():
        for country in tqdm(countries, desc=f"{cont}"):
            # Use soft-cleaned country for lookups and keys; brutal will be applied later globally
            lookup_name = country
            qid = wikidata_country_qid(lookup_name)
            if not qid:
                continue

            # ADM1 regions
            regs = wikidata_adm1_regions(qid)
            if regs:
                REGIONS_BY_COUNTRY.setdefault(lookup_name, [])
                lat_acc = lon_acc = 0.0
                nlat = nlon = 0
                for name, lat, lon in regs:
                    if not name:
                        continue
                    REGIONS_BY_COUNTRY[lookup_name].append(name)
                    if lat is not None and lon is not None:
                        REGION_CENTROIDS[name] = [lat, lon]
                        lat_acc += lat; nlat += 1
                        lon_acc += lon; nlon += 1
                if nlat and nlon:
                    COUNTRY_CENTROIDS[lookup_name] = [lat_acc/nlat, lon_acc/nlon]

            # Top-N cities
            try:
                cities = wikidata_top_cities(qid, top_n=args.top_cities, include_towns=args.include_towns)
            except Exception:
                cities = []

            if cities:
                CITIES_BY_COUNTRY.setdefault(lookup_name, [])
                for name, pop, lat, lon in cities:
                    if not name:
                        continue
                    CITIES_BY_COUNTRY[lookup_name].append(name)
                    CITY_COORDS[name] = [lat, lon]

            time.sleep(0.2)  # be kind to endpoints

    # 4) GLOBAL FINAL CLEANUP (brutal) — affects EVERYTHING
    out_CONTINENTS = clean_list(out_CONTINENTS)
    out_COUNTRIES = clean_dict_of_lists(out_COUNTRIES)

    # Re-key country-based dicts to cleaned country names
    def remap_by_country(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out = OrderedDict()
        for k, lst in d.items():
            kk = brutal_clean(k)
            if not kk:
                continue
            out.setdefault(kk, [])
            out[kk].extend(lst)
        # Dedup/clean lists
        for k in list(out.keys()):
            out[k] = clean_list(out[k])
        return out

    REGIONS_BY_COUNTRY = remap_by_country(REGIONS_BY_COUNTRY)
    CITIES_BY_COUNTRY  = remap_by_country(CITIES_BY_COUNTRY)

    # Clean coordinate dicts (keys only; values are floats)
    REGION_CENTROIDS = clean_coord_dict(REGION_CENTROIDS)
    CITY_COORDS      = clean_coord_dict(CITY_COORDS)

    # COUNTRY_CENTROIDS depends on country keys
    COUNTRY_CENTROIDS = clean_coord_dict(COUNTRY_CENTROIDS)

    # 5) Write payload
    payload = {
        "CONTINENTS": out_CONTINENTS,
        "COUNTRIES_BY_CONTINENT": out_COUNTRIES,
        "COUNTRY_CENTROIDS": COUNTRY_CENTROIDS,
        "REGIONS_BY_COUNTRY": REGIONS_BY_COUNTRY,
        "REGION_CENTROIDS": REGION_CENTROIDS,
        "CITIES_BY_COUNTRY": CITIES_BY_COUNTRY,
        "CITY_COORDS": CITY_COORDS,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {args.out}:")
    print(f"  Continents: {len(out_CONTINENTS)}")
    print(f"  Countries: {sum(len(v) for v in out_COUNTRIES.values())} (across {len(out_COUNTRIES)} continents)")
    print(f"  Countries w/ regions: {len(REGIONS_BY_COUNTRY)}")
    print(f"  Region labels: {len(REGION_CENTROIDS)} with coords")
    total_cities = sum(len(v) for v in CITIES_BY_COUNTRY.values())
    print(f"  Countries w/ top-{args.top_cities} cities: {len(CITIES_BY_COUNTRY)}")
    print(f"  City labels: {total_cities} with coords")

if __name__ == "__main__":
    main()
