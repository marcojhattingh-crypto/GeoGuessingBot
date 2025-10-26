# geo_tables.py — catalogs + centroids loader for StreetCLIP bot

from __future__ import annotations
import json, os
from typing import Dict, List, Tuple, Optional

# --------- Defaults (safe, minimal) ----------
CONTINENTS: List[str] = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]

COUNTRIES_BY_CONTINENT: Dict[str, List[str]] = {
    "Africa": [], "Asia": [], "Europe": [], "North America": [], "South America": [], "Oceania": []
}

CONTINENT_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "Africa": (-1.5, 17.5), "Asia": (34.0, 100.0), "Europe": (54.0, 15.0),
    "North America": (39.0, -98.0), "South America": (-14.0, -58.0), "Oceania": (-24.0, 135.0),
}

COUNTRY_CENTROIDS: Dict[str, Tuple[float, float]] = {}

REGIONS_BY_COUNTRY: Dict[str, List[str]] = {}
REGION_CENTROIDS: Dict[str, Tuple[float, float]] = {}

CITIES_BY_COUNTRY: Dict[str, List[str]] = {}
CITY_COORDS: Dict[str, Tuple[float, float]] = {}

# --------- Helpers ----------
def _merge_list(dst: List[str], src: List[str]) -> List[str]:
    seen = set(dst)
    for x in src:
        if x not in seen:
            dst.append(x); seen.add(x)
    return dst

def _merge_dict_of_lists(dst: Dict[str, List[str]], src: Dict[str, List[str]]) -> None:
    for k, v in src.items():
        dst[k] = _merge_list(dst.get(k, []), list(v))

def _merge_dict_of_coords(dst: Dict[str, Tuple[float, float]], src: Dict[str, List[float] | Tuple[float, float]]) -> None:
    for k, v in src.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                dst[k] = (float(v[0]), float(v[1]))
            except Exception:
                pass

# --------- Load/merge JSON if present ----------
def _load_json_if_exists():
    json_path = os.path.join(os.path.dirname(__file__), "geo_tables.json")
    if not os.path.isfile(json_path):
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    global CONTINENTS, COUNTRIES_BY_CONTINENT, COUNTRY_CENTROIDS
    global REGIONS_BY_COUNTRY, REGION_CENTROIDS, CITIES_BY_COUNTRY, CITY_COORDS

    if isinstance(data.get("CONTINENTS"), list) and data["CONTINENTS"]:
        # Keep our default order if possible; else replace
        base = [*CONTINENTS]
        CONTINENTS = _merge_list(base, data["CONTINENTS"])

    if isinstance(data.get("COUNTRIES_BY_CONTINENT"), dict):
        _merge_dict_of_lists(COUNTRIES_BY_CONTINENT, data["COUNTRIES_BY_CONTINENT"])

    if isinstance(data.get("COUNTRY_CENTROIDS"), dict):
        _merge_dict_of_coords(COUNTRY_CENTROIDS, data["COUNTRY_CENTROIDS"])

    if isinstance(data.get("REGIONS_BY_COUNTRY"), dict):
        _merge_dict_of_lists(REGIONS_BY_COUNTRY, data["REGIONS_BY_COUNTRY"])

    if isinstance(data.get("REGION_CENTROIDS"), dict):
        _merge_dict_of_coords(REGION_CENTROIDS, data["REGION_CENTROIDS"])

    if isinstance(data.get("CITIES_BY_COUNTRY"), dict):
        _merge_dict_of_lists(CITIES_BY_COUNTRY, data["CITIES_BY_COUNTRY"])

    if isinstance(data.get("CITY_COORDS"), dict):
        _merge_dict_of_coords(CITY_COORDS, data["CITY_COORDS"])

_load_json_if_exists()

# --------- Public convenience helpers ----------
def continent_of(country: str) -> Optional[str]:
    for cont, countries in COUNTRIES_BY_CONTINENT.items():
        if country in countries:
            return cont
    return None

def centroid_for(label: str, continent_hint: Optional[str] = None, country_hint: Optional[str] = None) -> Tuple[float, float]:
    if label in CITY_COORDS: return CITY_COORDS[label]
    if label in REGION_CENTROIDS: return REGION_CENTROIDS[label]
    if label in COUNTRY_CENTROIDS: return COUNTRY_CENTROIDS[label]
    if continent_hint and continent_hint in CONTINENT_CENTROIDS:
        return CONTINENT_CENTROIDS[continent_hint]
    return (0.0, 0.0)

# Explicit export list (helps IDEs & avoids import errors)
__all__ = [
    "CONTINENTS", "COUNTRIES_BY_CONTINENT", "COUNTRY_CENTROIDS", "CONTINENT_CENTROIDS",
    "REGIONS_BY_COUNTRY", "REGION_CENTROIDS", "CITIES_BY_COUNTRY", "CITY_COORDS",
    "continent_of", "centroid_for",
]
