# geograb_hotkeys_screen.py
# Hotkeys:
#   - Mouse4 (XButton1) or F6 = capture fullscreen, crop, keep 3 shots, merge into 1, delete the 3 frames
#   - Mouse5 (XButton2) or F7 = reset for next round, delete previous merged + any frames
#
# Output:
#   geograb_data/merged_latest.png  (what your Streamlit app should load)

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any
import json, time, threading, sys, re

from PIL import Image
from pynput import mouse, keyboard
import mss

BASE_DIR = Path("./geograb_data")
FRAMES_DIR = BASE_DIR / "frames"
STATE_JSON = BASE_DIR / "state.json"
CONFIG_JSON = BASE_DIR / "screen_crop.json"
MERGED_PATH = BASE_DIR / "merged_latest.png"

BASE_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Configuration ----------------
# Two crop modes supported:
#  1) margins_px: cut these many pixels from each edge (left/right/top/bottom)
#  2) relative:   a [0..1] rectangle within the screen: x0,y0,w,h
DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "margins_px",
    "margins_px": {"left": 20, "right": 20, "top": 140, "bottom": 200},
    # Example relative box for 16:9 monitors (tune if you prefer ratios):
    "relative": {"x0": 0.02, "y0": 0.12, "w": 0.96, "h": 0.76},
}

def load_config() -> Dict[str, Any]:
    if CONFIG_JSON.exists():
        try:
            cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            pass
    # Write default if missing to help users tweak later
    CONFIG_JSON.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return DEFAULT_CONFIG

CFG = load_config()

# ---------------- State ----------------
class State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.paths: List[str] = []
        self.round_idx = int(time.time())
        self.last_merged: Optional[str] = None

    def load(self):
        if STATE_JSON.exists():
            try:
                d = json.loads(STATE_JSON.read_text(encoding="utf-8"))
                self.count = int(d.get("count", 0))
                self.paths = list(d.get("paths", []))
                self.round_idx = int(d.get("round_idx", int(time.time())))
                self.last_merged = d.get("last_merged")
            except Exception:
                self.reset()

    def save(self):
        STATE_JSON.write_text(json.dumps({
            "count": self.count,
            "paths": self.paths,
            "round_idx": self.round_idx,
            "last_merged": self.last_merged,
            "merged_path": str(MERGED_PATH) if self.last_merged else None
        }, ensure_ascii=False, indent=2), encoding="utf-8")

STATE = State(); STATE.load()
LOCK = threading.Lock()

# ---------------- Capture & crop ----------------
def fullscreen_capture() -> Image.Image:
    with mss.mss() as sct:
        mon = sct.monitors[1]  # primary
        raw = sct.grab(mon)
        # Convert to PIL RGB
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        return img

def crop_image(img: Image.Image) -> Image.Image:
    mode = (CFG.get("mode") or "margins_px").lower()
    W, H = img.width, img.height
    if mode == "relative":
        r = CFG.get("relative", {})
        x0 = max(0.0, min(1.0, float(r.get("x0", 0.02))))
        y0 = max(0.0, min(1.0, float(r.get("y0", 0.12))))
        w  = max(0.0, min(1.0, float(r.get("w" , 0.96))))
        h  = max(0.0, min(1.0, float(r.get("h" , 0.76))))
        left   = int(W * x0)
        top    = int(H * y0)
        right  = int(min(W, left + W * w))
        bottom = int(min(H, top  + H * h))
        return img.crop((left, top, right, bottom))
    # default margins_px
    m = CFG.get("margins_px", {})
    left   = int(m.get("left", 20))
    right  = int(m.get("right", 20))
    top    = int(m.get("top", 140))
    bottom = int(m.get("bottom", 200))
    left   = max(0, left)
    top    = max(0, top)
    right  = max(0, right)
    bottom = max(0, bottom)
    return img.crop((left, top, W - right, H - bottom))

def merge_three(img_paths: List[Path], out_path: Path) -> Path:
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    min_h = min(im.height for im in imgs)
    resized = []
    for im in imgs:
        w = int(im.width * (min_h / im.height))
        resized.append(im.resize((w, min_h), Image.LANCZOS))
    total_w = sum(im.width for im in resized)
    canvas = Image.new("RGB", (total_w, min_h), (0, 0, 0))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(out_path)
    return out_path

def safe_delete(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

def cleanup_frames():
    # Delete all per-round frames
    for p in FRAMES_DIR.glob("*.png"):
        safe_delete(p)

def cleanup_merged():
    safe_delete(MERGED_PATH)

# ---------------- Actions ----------------
def action_capture():
    with LOCK:
        try:
            img = fullscreen_capture()
            img = crop_image(img)
            idx = STATE.count + 1
            fn = f"r{STATE.round_idx}_{idx:02d}.png"
            out = FRAMES_DIR / fn
            img.save(out)
            STATE.count = idx
            STATE.paths.append(str(out))
            print(f"[CAPTURE] {out}")
            if STATE.count == 3:
                merged = merge_three([Path(p) for p in STATE.paths[-3:]], MERGED_PATH)
                print(f"[MERGED] {merged}")
                # Immediately delete frames to save space
                cleanup_frames()
                STATE.paths = []
                STATE.last_merged = str(merged)
            STATE.save()
        except Exception as e:
            print(f"[ERROR] capture: {e}")

def action_next_round():
    with LOCK:
        # Delete any pending frames + the previous merged
        cleanup_frames()
        cleanup_merged()
        STATE.reset()
        STATE.save()
        print("[NEXT] reset done, files cleaned.")

# ---------------- Hotkey listeners ----------------
def on_mouse_click(x, y, button, pressed):
    if not pressed:
        return
    name = str(button).lower()
    # Some mice report Button.x1 / Button.x2
    if name.endswith(".x1"):
        action_capture()
    elif name.endswith(".x2"):
        action_next_round()

def on_key_press(key):
    # F6 = capture, F7 = next/reset (keyboard backup)
    try:
        if key == keyboard.Key.f6:
            action_capture()
        elif key == keyboard.Key.f7:
            action_next_round()
    except Exception:
        pass

def main():
    print("Hotkeys ready:")
    print("  Mouse4 (or F6): capture (3 shots → merge → delete frames)")
    print("  Mouse5 (or F7): next/reset (deletes merged + frames)")
    print(f"Output composite: {MERGED_PATH}")
    with mouse.Listener(on_click=on_mouse_click) as m_listener, \
         keyboard.Listener(on_press=on_key_press) as k_listener:
        try:
            m_listener.join()
            k_listener.join()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal: {e}")
        sys.exit(1)
