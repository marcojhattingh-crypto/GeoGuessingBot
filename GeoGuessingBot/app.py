# app.py — StreetCLIP Geo Bot (no Windows toasts; uses Streamlit toasts)
# Run:  streamlit run app.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io, json, time, threading
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st
from PIL import Image
try:
    from PIL import ImageGrab
    _IMAGEGRAB_AVAILABLE = True
except Exception:
    _IMAGEGRAB_AVAILABLE = False

# --- Optional/host-specific deps (guarded so cloud won't crash) ---
try:
    import mss
    HAVE_MSS = True
except Exception:
    HAVE_MSS = False

try:
    from pynput import keyboard
    HAVE_PYNPUT = True
except Exception:
    HAVE_PYNPUT = False

import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast, CLIPImageProcessor

import folium
from streamlit_folium import st_folium

# from streamlit.runtime.scriptrunner import add_script_run_ctx  # optional; guarded below

# ---- Your geo tables (unchanged) ----
from geo_tables import (
    CONTINENTS, COUNTRIES_BY_CONTINENT, CONTINENT_CENTROIDS,
    REGIONS_BY_COUNTRY, CITIES_BY_COUNTRY,
    centroid_for
)

# ---------- Streamlit page ----------
st.set_page_config(page_title="StreetCLIP — F6 screenshot + Streamlit toasts", layout="wide")

# ---------- Paths / persistence ----------
BASE_DIR   = Path("geograb_data")
BASE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_JSON = BASE_DIR / "screen_crop.json"
LAST_JPG    = BASE_DIR / "last_shot.jpg"

DEFAULT_CONFIG = {
    "mode": "margins_px",
    "margins_px": {"left": 20, "right": 20, "top": 140, "bottom": 200},
}

def load_config():
    if CONFIG_JSON.exists():
        try:
            cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            pass
    CONFIG_JSON.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return DEFAULT_CONFIG

def save_config(cfg):
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

CFG = load_config()

# ---------- Safe image open ----------
def safe_open_image(path: Path, attempts: int = 10, sleep: float = 0.1):
    last = -1
    stable = 0
    for _ in range(attempts):
        try:
            if not path.exists():
                time.sleep(sleep); continue
            sz = path.stat().st_size
            if sz == last and sz > 0:
                stable += 1
            else:
                stable = 0
            last = sz
            if stable >= 2:
                with open(path, "rb") as f:
                    data = f.read()
                img = Image.open(io.BytesIO(data))
                img.load()
                return img.convert("RGB")
        except Exception:
            pass
        time.sleep(sleep)
    return None

# ---------- Screenshot + crop ----------
def fullscreen_capture_crop(cfg) -> Image.Image:
    if not HAVE_MSS:
        raise RuntimeError("Screen capture (mss) is unavailable on this environment. Use file upload instead.")
    with mss.mss() as sct:
        mon = sct.monitors[1]
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.rgb)

    W, H = img.width, img.height
    m = (cfg.get("margins_px") or {})
    left   = max(0, int(m.get("left", 20)))
    right  = max(0, int(m.get("right", 20)))
    top    = max(0, int(m.get("top", 140)))
    bottom = max(0, int(m.get("bottom", 200)))
    return img.crop((left, top, W - right, H - bottom))

# ---------- Model / Processor ----------
MODEL_ID = "geolocal/StreetCLIP"
BASE_CLIP_PROCESSOR = "openai/clip-vit-large-patch14-336"

@st.cache_resource(show_spinner=True)
def load_model_and_processor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    image_proc = CLIPImageProcessor.from_pretrained(BASE_CLIP_PROCESSOR)
    tokenizer  = CLIPTokenizerFast.from_pretrained(BASE_CLIP_PROCESSOR, use_fast=True)
    processor  = CLIPProcessor(tokenizer=tokenizer, image_processor=image_proc)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    return model, processor, device

model, processor, device = load_model_and_processor()

# ---------- OLD scoring logic (restored) ----------
def score_prompts(image: Image.Image, prompts):
    texts = [f"a street scene in {p}" for p in prompts]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits_per_image[0]
        probs = logits.softmax(dim=0)
    return [(p, float(probs[i])) for i, p in enumerate(prompts)]

def predict_hier(image: Image.Image, topk_countries=5, topk_regions=6, topk_cities=8):
    cont_scores = score_prompts(image, CONTINENTS)
    cont_scores.sort(key=lambda x: x[1], reverse=True)
    continent = cont_scores[0][0] if cont_scores else None

    countries = []; best_country = None
    if continent:
        cand = COUNTRIES_BY_CONTINENT.get(continent, [])
        if cand:
            cs = score_prompts(image, cand)
            cs.sort(key=lambda x: x[1], reverse=True)
            countries = cs[:topk_countries]
            best_country = countries[0][0] if countries else None

    regions = []
    if best_country and best_country in REGIONS_BY_COUNTRY:
        rs = score_prompts(image, REGIONS_BY_COUNTRY[best_country])
        rs.sort(key=lambda x: x[1], reverse=True)
        regions = rs[:topk_regions]

    cities = []
    if best_country and best_country in CITIES_BY_COUNTRY:
        xs = score_prompts(image, CITIES_BY_COUNTRY[best_country])
        xs.sort(key=lambda x: x[1], reverse=True)
        cities = xs[:topk_cities]

    return {
        "continent": (continent, cont_scores),
        "countries": countries,
        "region_country": best_country,
        "regions": regions,
        "cities": cities,
    }

# ---------- Streamlit toasts (replacement for Windows toasts) ----------
_TOAST_LOCK = threading.Lock()
_LAST_TOAST_MS = 0.0
TOAST_COOLDOWN_MS = 600  # ms

def toast(lines):
    """Show a rate-limited toast inside Streamlit (cross-platform)."""
    global _LAST_TOAST_MS
    now = time.time() * 1000.0
    with _TOAST_LOCK:
        if now - _LAST_TOAST_MS < TOAST_COOLDOWN_MS:
            return
        _LAST_TOAST_MS = now
    try:
        msg = "\n".join(lines) if isinstance(lines, (list, tuple)) else str(lines)
        st.toast(msg, icon="🗺️")
    except Exception:
        # As a fallback, write to the page (non-blocking)
        st.info(msg)

def send_result_toast(pred):
    continent, cont_scores = pred.get("continent") or (None, [])
    countries = pred.get("countries") or []
    regions   = pred.get("regions") or []
    cities    = pred.get("cities") or []

    def fmt_top(lst): return f"{lst[0][0]} ({lst[0][1]*100:.1f}%)" if lst else "—"
    top_cont = f"{continent or '—'}" + (f" ({cont_scores[0][1]*100:.1f}%)" if cont_scores else "")
    toast([
        f"🌍 Continent: {top_cont}",
        f"🏳️ Country: {fmt_top(countries)}",
        f"🗺️ Region: {fmt_top(regions)}",
        f"🏙️ City: {fmt_top(cities)}",
    ])

# ---------- Runtime state / debouncing ----------
_RUNTIME_LOCK = threading.Lock()
_PROCESSING = threading.Lock()     # one inference at a time
_LAST_TIME: Optional[float] = None
_LAST_PRED = None
_LAST_F6_MS = 0.0
DEBOUNCE_MS = 1200                 # bigger window to kill auto-repeat

def _handle_capture_and_predict():
    if not _PROCESSING.acquire(blocking=False):
        return
    try:
        img = fullscreen_capture_crop(CFG)
        try:
            LAST_JPG.parent.mkdir(parents=True, exist_ok=True)
            img.save(LAST_JPG, quality=92)
        except Exception:
            pass

        toast(["📸 Captured screenshot", "Processing image…"])
        pred = predict_hier(img, 5, 6, 8)

        with _RUNTIME_LOCK:
            global _LAST_TIME, _LAST_PRED
            _LAST_TIME = time.time()
            _LAST_PRED = pred

        send_result_toast(pred)
    except Exception as e:
        toast([f"Capture failed: {e!r}"])
    finally:
        _PROCESSING.release()

def _on_key_press(key):
    global _LAST_F6_MS
    try:
        if getattr(key, "vk", None) == 0x75 or str(key) in ("Key.f6", "<96>"):  # robust F6 detect
            now = time.time() * 1000.0
            if now - _LAST_F6_MS < DEBOUNCE_MS:
                return
            _LAST_F6_MS = now
            _handle_capture_and_predict()
    except Exception as e:
        toast([f"Hotkey error: {e!r}"])

# ---------- Listener: make it a session singleton ----------
def _start_listener_once():
    if not HAVE_PYNPUT:
        st.warning("Hotkey listener (pynput) isn’t available here. Use the **Upload** button below to analyze images.")
        return

    if "f6_thread" in st.session_state:
        t = st.session_state["f6_thread"]
        if isinstance(t, threading.Thread) and t.is_alive():
            return  # already running

    def _keyboard_loop():
        try:
            with keyboard.Listener(on_press=_on_key_press) as l:
                l.join()
        except Exception as e:
            toast([f"Listener stopped: {e!r}"])

    t = threading.Thread(target=_keyboard_loop, daemon=True, name="F6Listener")
    try:
        # Late import so it doesn't explode on older runtimes
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(t)
    except Exception:
        pass
    t.start()
    st.session_state["f6_thread"] = t

_start_listener_once()

# ---------- UI ----------
st.title("🗺️ StreetCLIP — press F6 (local) or upload (cloud)")

with st.expander("📐 Screenshot crop (margins in pixels)", expanded=True):
    m = CFG.get("margins_px", {})
    c1, c2, c3, c4 = st.columns(4)
    left  = c1.number_input("Left", 0, 2000, int(m.get("left", 20)))
    right = c2.number_input("Right", 0, 2000, int(m.get("right", 20)))
    top   = c3.number_input("Top", 0, 2000, int(m.get("top", 140)))
    bot   = c4.number_input("Bottom", 0, 2000, int(m.get("bottom", 200)))
    if st.button("Save margins"):
        CFG["margins_px"] = {"left": left, "right": right, "top": top, "bottom": bot}
        save_config(CFG)
        st.success("Saved. On desktop: focus your game and press F6. On cloud: use Upload below.")

with _RUNTIME_LOCK:
    ltime = _LAST_TIME
    pred = _LAST_PRED

st.subheader("Status")
colA, colB, colC = st.columns(3)
colA.metric("Hotkey listener", "Running" if HAVE_PYNPUT else "Unavailable")
colB.metric("Last capture", "—" if not ltime else datetime.fromtimestamp(ltime).strftime("%H:%M:%S"))
colC.metric("Saved image", LAST_JPG.name if LAST_JPG.exists() else "—")

# --- Cloud-friendly fallback: upload an image to analyze ---
st.markdown("### 📤 Upload image (cloud-friendly)")
up = st.file_uploader("Upload a screenshot to analyze", type=["png", "jpg", "jpeg"])
if up is not None:
    try:
        img_u = Image.open(up).convert("RGB")
        pred_u = predict_hier(img_u, 5, 6, 8)
        _LAST_PRED = pred_u
        _LAST_TIME = time.time()
        st.success("Uploaded image processed.")
        send_result_toast(pred_u)
        img_u.save(LAST_JPG, quality=92)
    except Exception as e:
        st.error(f"Failed to process uploaded image: {e!r}")

left, right = st.columns([0.46, 0.54], gap="large")

with left:
    st.write("**Last screenshot / uploaded image**")
    if LAST_JPG.exists():
        img = safe_open_image(LAST_JPG)
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.caption("…writing image, refresh in a moment.")
    else:
        if HAVE_MSS:
            st.info("No image yet. Focus your game and press **F6**.")
        else:
            st.info("No image yet. Use **Upload** above to analyze an image.")

with right:
    st.write("**Last prediction** (hierarchical)")
    if _LAST_PRED:
        continent, cont_scores = _LAST_PRED["continent"]
        st.write("🌍 **Continent**", {lbl: round(p*100,2) for lbl, p in cont_scores})
        if _LAST_PRED["countries"]:
            st.write("🏳️ **Countries (top-5)**", {lbl: round(p*100,2) for lbl, p in _LAST_PRED["countries"]})
        if _LAST_PRED["regions"]:
            st.write("🗺️ **Regions (top-6)**", {lbl: round(p*100,2) for lbl, p in _LAST_PRED["regions"]})
        if _LAST_PRED["cities"]:
            st.write("🏙️ **Cities (top-8)**", {lbl: round(p*100,2) for lbl, p in _LAST_PRED["cities"]})

# Map pin for best guess
m = folium.Map(location=[0, 0], zoom_start=2)
pred_map = _LAST_PRED
if pred_map:
    cont = pred_map["continent"][0] if pred_map["continent"] else None
    best_country = pred_map["region_country"]
    best_city = pred_map["cities"][0][0] if pred_map["cities"] else None
    best_region = pred_map["regions"][0][0] if pred_map["regions"] else None
    if best_city:
        label = best_city
        lat, lon = centroid_for(best_city, continent_hint=cont, country_hint=best_country)
        zoom = 8
    elif best_region:
        label = best_region
        lat, lon = centroid_for(best_region, continent_hint=cont, country_hint=best_country)
        zoom = 6
    elif best_country:
        label = best_country
        lat, lon = centroid_for(best_country, continent_hint=cont)
        zoom = 4
    elif cont:
        label = cont
        lat, lon = CONTINENT_CENTROIDS.get(cont, (0.0, 0.0))
        zoom = 3
    else:
        label = "—"; lat, lon = 0.0, 0.0; zoom = 2
    folium.Marker([lat, lon], tooltip=label, popup=f"Top pin: {label}").add_to(m)
    m.location = [lat, lon]; m.zoom_start = zoom

st_folium(m, width=900, height=560)

st.caption("Keep this app running. On desktop: focus your game and press **F6**. On cloud: upload an image. Notifications are shown as Streamlit toasts in the bottom-right.")
