# vlm_step_drive_live_v8_4_SEARCH_REACT_FAST_NO_BACK.py
from __future__ import annotations

import base64, json, math, queue, re, threading, time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import requests
import cv2

# -----------------------------
# USER CONFIG
# -----------------------------
ROBOT_BASE_URL = "http://192.168.1.177:5000"
MJPEG_URL      = f"{ROBOT_BASE_URL}/video_feed"
CMD_URL        = f"{ROBOT_BASE_URL}/cmd"

OLLAMA_URL   = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5vl:3b"

VLM_ENABLED_DEFAULT = True

# -----------------------------
# UI
# -----------------------------
WINDOW_NAME = "Vision"
FONT_SCALE_DEFAULT = 0.55
FONT_THICKNESS = 1
LINE_H = 18

# -----------------------------
# Sharpness gating
# -----------------------------
SHARP_TARGET_STILL = 16.0   # "идеально" для CV/VLM
SHARP_MIN_VLM      = 11.0   # минимально допустимо для VLM на плохом свете
SHARP_MIN_TRACK    = 10.0

# -----------------------------
# HSV (STRICT for ball, LOOSE for "yellow trigger")
# -----------------------------
# STRICT (под мяч) — можно подстроить, но не делай слишком узко
HSV_Y_LO_STRICT = (15, 60, 60)
HSV_Y_HI_STRICT = (55, 255, 255)

# LOOSE — только чтобы понять: "в кадре есть большой жёлтый объект"
HSV_Y_LO_LOOSE = (10, 40, 40)
HSV_Y_HI_LOOSE = (65, 255, 255)

# -----------------------------
# CV STRICT constraints (чуть мягче, чем было)
# -----------------------------
MIN_AREA_FRAC_CAND = 0.003
MAX_AREA_FRAC_CAND = 0.55
MIN_CIRC = 0.42
MIN_SOLIDITY = 0.84
MAX_ASPECT_DEV = 0.70
MIN_DARK_RATIO = 0.018
MIN_DARK_BLOBS = 1

# "супер-уверенный CV" может вести даже при включенном VLM (чтобы не игнорировать явный мяч)
CV_STRONG_SCORE = 0.82
CV_STRONG_AREA  = 0.020

# Center deadzone
DEADZONE_CX = 0.06
CX_SMOOTH_ALPHA = 0.45

# -----------------
# SEARCH (stop-and-analyze, реагирует быстро)
# -----------------
SEARCH_ROT_SEC          = 0.12   # меньше — меньше "пролетает мимо"
SEARCH_SETTLE_SEC       = 0.18
SEARCH_STILL_STABLE_SEC = 0.10
SEARCH_MAX_STOP_SEC     = 1.20   # даём VLM спокойно отработать на стопе

SEARCH_SWEEP_STEPS = 60
ONE_DIR_SCAN_DEFAULT = True

# Stop keepalive (если иногда "stop" теряется/не доходит)
STOP_KEEPALIVE_SEC = 0.18

# -----------------
# TRACK
# -----------------
TRACK_ROT_SEC_COARSE = 0.055
TRACK_ROT_SEC_FINE   = 0.038
TRACK_FWD_SEC_FAR    = 0.22
TRACK_FWD_SEC_NEAR   = 0.10
TRACK_STAB_SEC       = 0.14

MAX_ROT_STREAK = 7
LOSS_HOLD_SEC = 0.70

# -----------------
# VLM cadence
# -----------------
ANALYZE_MAX_FRAMEAGE  = 0.25
ANALYZE_MIN_INTERVAL  = 0.22
VLM_TIMEOUT_SEC       = 4.2

# В SEARCH: проверяем часто на стопе (иначе мяч в кадре, а мы сканируем дальше)
SCAN_VLM_PERIOD_SEC   = 0.80
SCAN_VLM_COOLDOWN_SEC = 0.55

# Goal detection (строго + центр)
GOAL_AREA_FRAC = 0.38
GOAL_CONFIRM_N = 4
GOAL_REQUIRE_CENTERED = True
CLOSE_MIN_AREA_SUPPORT = 0.20
VLM_AREA_MAX_SANE = 0.90

FLIP_X_DEFAULT = False

# CV history (обновляем только на стоячих резких кадрах)
CV_CONFIRM_WIN = 5
CV_CONFIRM_MIN = 3
CV_SUPPRESS_AFTER_VLM_NO_SEC = 1.2

# Optional voice
try:
    import pyttsx3
    HAS_TTS = True
except Exception:
    HAS_TTS = False


def now() -> float:
    return time.time()

def clamp(x, a, b):
    return a if x < a else b if x > b else x

def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def jpeg_bytes_from_bgr(frame_bgr: np.ndarray, quality=85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""

def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    if "```" in t:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, flags=re.S | re.I)
        if m:
            t = m.group(1).strip()
        else:
            t = t.replace("```json", "").replace("```", "").strip()
    t = re.sub(r"^\s*json\s*", "", t, flags=re.I)

    try:
        j = json.loads(t)
        return j if isinstance(j, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    chunk = m.group(0)
    chunk = chunk[: chunk.rfind("}") + 1]
    try:
        j = json.loads(chunk)
        return j if isinstance(j, dict) else None
    except Exception:
        return None


# -----------------------------
# Robot commander (с force для stop keepalive)
# -----------------------------
class RobotCommander:
    def __init__(self, cmd_url: str):
        self.cmd_url = cmd_url
        self.last_cmd = "stop"
        self.last_cmd_ts = 0.0

    def send(self, a: str, timeout=0.35, force: bool = False):
        t = now()
        if (not force) and a == self.last_cmd and (t - self.last_cmd_ts) < 0.03:
            return
        try:
            requests.get(self.cmd_url, params={"a": a}, timeout=timeout)
            self.last_cmd = a
            self.last_cmd_ts = t
        except Exception:
            # не обновляем last_cmd/ts, чтобы следующий вызов попробовал ещё раз
            pass

    def stop(self, force: bool = False):
        self.send("stop", force=force)


class MotionScheduler:
    def __init__(self, commander: RobotCommander):
        self.cmdr = commander
        self.active = False
        self.cmd = "stop"
        self.t_end_cmd = 0.0
        self.t_end_stab = 0.0
        self.last_motion_end_ts = 0.0
        self.last_pulse_cmd = "stop"
        self.last_pulse_ts = 0.0

    def busy(self) -> bool:
        return self.active

    def cancel(self):
        self.active = False
        self.cmd = "stop"
        self.t_end_cmd = 0.0
        self.t_end_stab = 0.0
        self.cmdr.stop(force=True)
        self.last_motion_end_ts = now()

    def start_pulse(self, cmd: str, dur: float, stab: float):
        if self.active:
            return
        dur = max(0.0, float(dur))
        stab = max(0.0, float(stab))
        t = now()
        self.active = True
        self.cmd = cmd
        self.t_end_cmd = t + dur
        self.t_end_stab = t + dur + stab
        self.last_pulse_cmd = cmd
        self.last_pulse_ts = t
        self.cmdr.send(cmd)

    def update(self):
        if not self.active:
            return
        t = now()
        if t >= self.t_end_cmd and self.cmd != "stop":
            self.cmd = "stop"
            # усиленный stop
            self.cmdr.stop(force=True)
        if t >= self.t_end_stab:
            self.active = False
            self.last_motion_end_ts = t


# -----------------------------
# MJPEG grabber
# -----------------------------
class MJPEGFrameGrabber:
    def __init__(self, mjpeg_url: str, timeout=3.0):
        self.url = mjpeg_url
        self.timeout = timeout
        self._lock = threading.Lock()
        self._latest = None  # (frame_bgr, ts)
        self._alive = False
        self._thread = None
        self._stop = threading.Event()
        self._stats_fps = 0.0
        self._fps_cnt = 0
        self._fps_t0 = now()

    def start(self):
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._alive = False

    def alive(self) -> bool:
        return self._alive

    def get_latest(self) -> Optional[Tuple[np.ndarray, float]]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest[0].copy(), self._latest[1]

    def fps(self) -> float:
        return float(self._stats_fps)

    def _run(self):
        buf = b""
        session = requests.Session()
        headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
        while not self._stop.is_set():
            try:
                with session.get(self.url, stream=True, timeout=self.timeout, headers=headers) as r:
                    r.raise_for_status()
                    buf = b""
                    for chunk in r.iter_content(chunk_size=4096):
                        if self._stop.is_set():
                            break
                        if not chunk:
                            continue
                        buf += chunk
                        if len(buf) > 2_000_000:
                            buf = buf[-500_000:]
                        a = buf.find(b"\xff\xd8")
                        b = buf.find(b"\xff\xd9")
                        if a != -1 and b != -1 and b > a:
                            jpg = buf[a:b+2]
                            buf = buf[b+2:]
                            arr = np.frombuffer(jpg, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is None:
                                continue
                            ts = now()
                            with self._lock:
                                self._latest = (frame, ts)

                            self._fps_cnt += 1
                            dt = ts - self._fps_t0
                            if dt >= 1.0:
                                self._stats_fps = self._fps_cnt / dt
                                self._fps_cnt = 0
                                self._fps_t0 = ts
                time.sleep(0.10)
            except Exception:
                self._alive = False
                time.sleep(0.25)
                self._alive = True


# -----------------------------
# CV detection (LOOSE + STRICT)
# -----------------------------
@dataclass
class CVCand:
    cand: bool = False
    cx: float = 0.5
    area_frac: float = 0.0
    score: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    note: str = ""

def _circularity(area: float, perim: float) -> float:
    if perim <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))

def _solidity(cnt) -> float:
    area = float(cv2.contourArea(cnt))
    if area <= 1e-6:
        return 0.0
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1e-6:
        return 0.0
    return float(area / hull_area)

def _dark_blob_check(frame_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[float,int]:
    x,y,w,h = bbox
    H, W = frame_bgr.shape[:2]
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    roi = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0, 0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = int(np.percentile(gray, 25))
    thr = int(clamp(thr, 25, 115))
    _, m = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    dark_ratio = float(np.count_nonzero(m)) / float(m.size)

    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= 0.0020 * (w*h):
            blobs += 1

    return dark_ratio, blobs

def cv_detect_yellow_loose(frame_bgr: np.ndarray) -> CVCand:
    """Дешёвый триггер: есть ли крупный жёлтый объект в кадре (без проверки чёрных пятен)."""
    H, W = frame_bgr.shape[:2]
    area_frame = float(H * W)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(HSV_Y_LO_LOOSE, np.uint8), np.array(HSV_Y_HI_LOOSE, np.uint8))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = CVCand(False, 0.5, 0.0, 0.0, None, "")
    if not cnts:
        return best

    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < 50:
            continue
        area_frac = a / area_frame
        if area_frac < 0.004 or area_frac > 0.70:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cx = (x + w/2.0) / float(W)
        score = clamp(area_frac / 0.20, 0.0, 1.0)
        if score > best.score:
            best = CVCand(True, cx, area_frac, score, (x,y,w,h), "loose_yellow")
    return best

def cv_detect_ball_strict(frame_bgr: np.ndarray) -> CVCand:
    H, W = frame_bgr.shape[:2]
    area_frame = float(H * W)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(HSV_Y_LO_STRICT, np.uint8), np.array(HSV_Y_HI_STRICT, np.uint8))

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = CVCand(False, 0.5, 0.0, 0.0, None, "")
    if not cnts:
        return best

    for c in cnts:
        a = float(cv2.contourArea(c))
        if a <= 10:
            continue
        area_frac = a / area_frame
        if area_frac < MIN_AREA_FRAC_CAND or area_frac > MAX_AREA_FRAC_CAND:
            continue

        per = float(cv2.arcLength(c, True))
        circ = _circularity(a, per)
        if circ < MIN_CIRC:
            continue

        x,y,w,h = cv2.boundingRect(c)
        if w <= 1 or h <= 1:
            continue
        aspect = float(w) / float(h)
        if abs(aspect - 1.0) > MAX_ASPECT_DEV:
            continue

        sol = _solidity(c)
        if sol < MIN_SOLIDITY:
            continue

        dark_ratio, blobs = _dark_blob_check(frame_bgr, (x,y,w,h))
        if dark_ratio < MIN_DARK_RATIO or blobs < MIN_DARK_BLOBS:
            continue

        cx = (x + w/2.0) / float(W)

        score = (
            0.55 * clamp(area_frac / 0.20, 0.0, 1.0) +
            0.20 * clamp((circ - MIN_CIRC) / (1.0 - MIN_CIRC), 0.0, 1.0) +
            0.15 * clamp((dark_ratio - MIN_DARK_RATIO) / 0.12, 0.0, 1.0) +
            0.10 * clamp((sol - MIN_SOLIDITY) / (1.0 - MIN_SOLIDITY), 0.0, 1.0)
        )

        if score > best.score:
            best = CVCand(True, cx, area_frac, score, (x,y,w,h),
                          f"circ={circ:.2f} sol={sol:.2f} dark={dark_ratio:.3f} blobs={blobs}")
    return best


# -----------------------------
# VLM client
# -----------------------------
@dataclass
class VLMState:
    alive: bool = False
    ok: bool = False
    last_ts: float = 0.0
    last_raw: str = ""
    last_json: Optional[Dict[str, Any]] = None
    busy: bool = False
    err: str = ""

class OllamaVLM:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model
        self.state = VLMState(alive=False, ok=False, last_ts=0.0, last_raw="", last_json=None, busy=False, err="")
        self._lock = threading.Lock()
        self.last_good_ts = 0.0
        self.last_good_json: Optional[Dict[str, Any]] = None
        self.last_good_raw: str = ""
        self.last_no_ball_ts = 0.0

    def ask_sync(self, frame_bgr: np.ndarray, timeout: float = 2.0) -> VLMState:
        with self._lock:
            if self.state.busy:
                return self.state
            self.state.busy = True
            self.state.err = ""

        img_jpg = jpeg_bytes_from_bgr(frame_bgr, quality=82)
        img_b64 = base64.b64encode(img_jpg).decode("ascii")

        prompt = (
            "Answer ONLY in JSON. No extra text.\n"
            "Detect the YELLOW soccer ball with BLACK pentagons.\n"
            "Return exactly:\n"
            "{"
            "\"ball\": true/false, "
            "\"cx\": number (0..1 center x OR pixel x), "
            "\"area\": number (0..1 fraction of image area), "
            "\"close\": true/false"
            "}\n"
            "If not sure: ball=false, cx=0.5, area=0.0, close=false.\n"
        )

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Return only JSON, no extra text."},
                {"role": "user", "content": prompt, "images": [img_b64]},
            ],
            "options": {"temperature": 0.0},
        }

        st = VLMState(alive=True, ok=False, last_ts=now(), last_raw="", last_json=None, busy=True, err="")
        try:
            r = requests.post(self.url, json=payload, timeout=timeout)
            if r.status_code != 200:
                st.err = f"HTTP {r.status_code}"
            else:
                data = r.json()
                content = data.get("message", {}).get("content", "")
                st.last_raw = content
                j = safe_json_extract(content)
                if isinstance(j, dict) and ("ball" in j):
                    st.last_json = j
                    st.ok = True
                else:
                    st.err = "bad_json"
        except Exception as e:
            st.err = f"{type(e).__name__}"

        st.busy = False

        with self._lock:
            self.state = st
            if st.ok and isinstance(st.last_json, dict):
                self.last_good_ts = st.last_ts
                self.last_good_json = st.last_json
                self.last_good_raw = st.last_raw
                try:
                    if bool(st.last_good_json.get("ball", False)) is False:
                        self.last_no_ball_ts = st.last_ts
                except Exception:
                    pass
            self.state.busy = False

        return st

    def get_state(self) -> VLMState:
        with self._lock:
            return self.state

    def get_last_good(self) -> Tuple[float, Optional[Dict[str, Any]], str]:
        with self._lock:
            return self.last_good_ts, self.last_good_json, self.last_good_raw

    def get_last_no_ball_ts(self) -> float:
        with self._lock:
            return float(self.last_no_ball_ts)


# -----------------------------
# Voice
# -----------------------------
class Voice:
    def __init__(self):
        self.enabled = False
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._engine = None
        if HAS_TTS:
            self._engine = pyttsx3.init()

    def start(self):
        if not HAS_TTS:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def say(self, text: str):
        if not (HAS_TTS and self.enabled):
            return
        try:
            self._q.put_nowait(text)
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception:
                pass


# -----------------------------
# Helpers
# -----------------------------
def decide_centered(cx: float) -> bool:
    return abs(cx - 0.5) <= DEADZONE_CX

def overlay_lines(img: np.ndarray, lines, font_scale=0.55):
    y = 18
    for s in lines:
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255,255,255), FONT_THICKNESS, cv2.LINE_AA)
        y += LINE_H

def draw_bbox(img: np.ndarray, bbox, color=(0,165,255), thickness=2):
    if bbox is None:
        return
    x,y,w,h = bbox
    cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness)

def parse_vlm_ball(j: Dict[str, Any], W: int) -> Tuple[bool, float, float, bool]:
    ball = bool(j.get("ball", False))
    close = bool(j.get("close", False))
    cx = float(j.get("cx", 0.5))
    area = float(j.get("area", 0.0))

    if cx > 1.5 and W > 10:
        cx = cx / float(W)

    cx = clamp(cx, 0.0, 1.0)
    area = clamp(area, 0.0, 1.0)
    if area > VLM_AREA_MAX_SANE:
        area = VLM_AREA_MAX_SANE
    return ball, cx, area, close


# -----------------------------
# Main
# -----------------------------
def main():
    auto = True
    cv_on = True
    flip_x = FLIP_X_DEFAULT
    vlm_enabled = VLM_ENABLED_DEFAULT
    one_dir_scan = ONE_DIR_SCAN_DEFAULT
    font_scale = FONT_SCALE_DEFAULT

    commander = RobotCommander(CMD_URL)
    motion = MotionScheduler(commander)

    grabber = MJPEGFrameGrabber(MJPEG_URL)
    grabber.start()

    vlm = OllamaVLM(OLLAMA_URL, OLLAMA_MODEL)
    voice = Voice()
    voice.start()
    voice.enabled = True if HAS_TTS else False

    mode = "SEARCH"  # SEARCH | TRACK
    sweep_dir = -1
    sweep_step = 0
    rot_streak = 0

    cx_smooth = 0.5
    last_seen_ts = 0.0
    last_seen_cx = 0.5
    last_seen_area = 0.0
    last_seen_src = "none"

    goal_hits = 0
    at_goal_now = False
    at_goal_prev = False

    rot_left_cmd = "rot_l"
    rot_right_cmd = "rot_r"

    cv_hist = [0] * CV_CONFIRM_WIN
    cv_hist_i = 0
    cv_confirmed = False
    cv_sum = 0

    last_analyze_ts = 0.0
    last_scan_vlm_ts = 0.0

    # SEARCH stop-and-analyze
    search_waiting = False
    search_wait_t0 = 0.0
    sharp_good_t0 = 0.0

    last_stop_keepalive_ts = 0.0

    speak_mem: Dict[str, float] = {}
    def speak_once(tag: str, text: str, cooldown=1.2):
        tt = now()
        if (tt - speak_mem.get(tag, 0.0)) > cooldown:
            speak_mem[tag] = tt
            voice.say(text)

    def stopped_settled() -> bool:
        return (not motion.busy()) and ((now() - motion.last_motion_end_ts) >= SEARCH_SETTLE_SEC)

    def frame_recent(frame_ts: float) -> bool:
        return (now() - frame_ts) <= ANALYZE_MAX_FRAMEAGE

    def stable_for_vlm(frame_ts: float, sharp: float) -> bool:
        if not frame_recent(frame_ts):
            return False
        if not stopped_settled():
            return False
        if sharp < SHARP_MIN_VLM:
            return False
        if (now() - last_analyze_ts) < ANALYZE_MIN_INTERVAL:
            return False
        return True

    def stop_keepalive():
        nonlocal last_stop_keepalive_ts
        t = now()
        if (t - last_stop_keepalive_ts) >= STOP_KEEPALIVE_SEC:
            commander.stop(force=True)
            last_stop_keepalive_ts = t

    print("Controls: Space(AUTO)  b(CV)  v(VLM now)  t(voice)  m(flipX)  o(oneDirScan)  +/- font  q/Esc quit")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fid_cnt = 0
    t0 = now()
    fps_local = 0.0

    while True:
        motion.update()

        got = grabber.get_latest()
        if got is None:
            commander.stop(force=True)
            blank = np.zeros((360,640,3), np.uint8)
            overlay_lines(blank, ["No MJPEG frames...", f"url: {MJPEG_URL}"], font_scale)
            cv2.imshow(WINDOW_NAME, blank)
            k = cv2.waitKey(10) & 0xFF
            if k in (27, ord('q')):
                break
            continue

        frame, frame_ts = got
        fid_cnt += 1

        if flip_x:
            frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = laplacian_sharpness(gray)

        dt = now() - t0
        if dt >= 1.0:
            fps_local = fid_cnt / dt
            fid_cnt = 0
            t0 = now()

        still = stopped_settled()

        # CV LOOSE trigger (на стопе, можно даже при чуть меньшей резкости)
        cv_loose = CVCand()
        if cv_on and still and sharp >= SHARP_MIN_VLM:
            cv_loose = cv_detect_yellow_loose(frame)

        # CV STRICT (только на реально резком кадре)
        cv_obs = CVCand()
        if cv_on and still and sharp >= SHARP_TARGET_STILL:
            cv_obs = cv_detect_ball_strict(frame)
            cv_hist[cv_hist_i] = 1 if cv_obs.cand else 0
            cv_hist_i = (cv_hist_i + 1) % CV_CONFIRM_WIN
            cv_sum = sum(cv_hist)
            cv_confirmed = (cv_sum >= CV_CONFIRM_MIN)
        else:
            cv_hist = [0] * CV_CONFIRM_WIN
            cv_hist_i = 0
            cv_sum = 0
            cv_confirmed = False

        # VLM state
        vst = vlm.get_state()
        v_age = (now() - vst.last_ts) if vst.last_ts > 0 else 9999.0

        # last good VLM
        good_ts, good_j, good_raw = vlm.get_last_good()
        good_age = (now() - good_ts) if good_ts > 0 else 9999.0

        vlm_ball = False
        vlm_cx = 0.5
        vlm_area = 0.0
        vlm_close = False
        vlm_good_recent = False

        if isinstance(good_j, dict) and good_age < 2.0:
            try:
                vlm_ball, vlm_cx, vlm_area, vlm_close = parse_vlm_ball(good_j, W)
                vlm_good_recent = True
            except Exception:
                vlm_good_recent = False

        # suppression (только для "CV как seen", но НЕ для loose-trigger)
        suppress_cv = False
        t_no = vlm.get_last_no_ball_ts()
        if t_no > 0 and (now() - t_no) < CV_SUPPRESS_AFTER_VLM_NO_SEC:
            suppress_cv = True

        # -----------------
        # SEARCH: стоп между поворотами + быстрый VLM при наличии жёлтого
        # -----------------
        if auto and mode == "SEARCH":
            if search_waiting:
                # держим стоп жёстко
                stop_keepalive()

                # резкость стабилизировалась?
                if still and sharp >= SHARP_TARGET_STILL:
                    if sharp_good_t0 <= 0:
                        sharp_good_t0 = now()
                else:
                    sharp_good_t0 = 0.0

                stable_sharp = (sharp_good_t0 > 0 and (now() - sharp_good_t0) >= SEARCH_STILL_STABLE_SEC)
                waited_long = ((now() - search_wait_t0) >= SEARCH_MAX_STOP_SEC)

                # КЛЮЧЕВОЕ: если в кадре есть крупный жёлтый объект — срочно спросить VLM.
                want_vlm = False
                if vlm_enabled and (not vst.busy) and stable_for_vlm(frame_ts, sharp):
                    if cv_loose.cand and cv_loose.area_frac >= 0.010:
                        want_vlm = True
                    elif (now() - last_scan_vlm_ts) >= SCAN_VLM_PERIOD_SEC and (stable_sharp or waited_long):
                        want_vlm = True

                    if want_vlm and (now() - last_scan_vlm_ts) >= SCAN_VLM_COOLDOWN_SEC:
                        last_analyze_ts = now()
                        last_scan_vlm_ts = now()
                        vlm.ask_sync(frame, timeout=VLM_TIMEOUT_SEC)

                # выход из ожидания: если нет большого жёлтого (или уже долго стоим)
                # если есть крупный жёлтый, НЕ крутим дальше, пока VLM не ответит (иначе "пролетает мимо")
                if (cv_loose.cand and cv_loose.area_frac >= 0.010):
                    # держим стоп (не выходим), чтобы не терять мяч во время анализа
                    pass
                else:
                    if stable_sharp or waited_long:
                        search_waiting = False
                        sharp_good_t0 = 0.0

            else:
                # следующий шаг сканирования
                if not motion.busy():
                    scan_cmd = rot_left_cmd if (one_dir_scan or sweep_dir < 0) else rot_right_cmd
                    motion.start_pulse(scan_cmd, SEARCH_ROT_SEC, 0.0)
                    sweep_step += 1
                    if sweep_step >= SEARCH_SWEEP_STEPS:
                        sweep_step = 0
                        if not one_dir_scan:
                            sweep_dir *= -1
                    search_waiting = True
                    search_wait_t0 = now()
                    sharp_good_t0 = 0.0

        # -----------------
        # Seen decision
        # -----------------
        seen = False
        src = "none"
        use_cx = 0.5
        use_area = 0.0

        # 1) VLM — главный
        if vlm_good_recent and vlm_ball:
            seen = True
            src = "vlm"
            use_cx = vlm_cx
            use_area = vlm_area

        # 2) CV strong — чтобы не игнорировать очевидный мяч
        if (not seen) and (cv_confirmed and cv_obs.cand and (not suppress_cv) and still and sharp >= SHARP_TARGET_STILL):
            if cv_obs.score >= CV_STRONG_SCORE and cv_obs.area_frac >= CV_STRONG_AREA:
                seen = True
                src = "cv_strong"
                use_cx = cv_obs.cx
                use_area = cv_obs.area_frac

        # smooth / hold
        if seen:
            cx_smooth = (1.0 - CX_SMOOTH_ALPHA) * cx_smooth + CX_SMOOTH_ALPHA * use_cx
            last_seen_ts = now()
            last_seen_cx = cx_smooth
            last_seen_area = use_area
            last_seen_src = src
        else:
            if (now() - last_seen_ts) < LOSS_HOLD_SEC:
                cx_smooth = (1.0 - CX_SMOOTH_ALPHA) * cx_smooth + CX_SMOOTH_ALPHA * last_seen_cx
            else:
                cx_smooth = 0.5

        centered_s = decide_centered(cx_smooth)

        # -----------------
        # GOAL logic
        # -----------------
        at_goal_prev = at_goal_now
        at_goal_now = False

        if vlm_close:
            cv_support = cv_obs.area_frac if (cv_confirmed and cv_obs.cand) else 0.0
            if max(vlm_area, cv_support) < CLOSE_MIN_AREA_SUPPORT:
                vlm_close = False

        if seen:
            big_now = (use_area >= GOAL_AREA_FRAC)
            close_now = (src == "vlm" and vlm_close and max(use_area, (cv_obs.area_frac if cv_confirmed else 0.0)) >= CLOSE_MIN_AREA_SUPPORT)
            centered_req_ok = (centered_s if GOAL_REQUIRE_CENTERED else True)
            if (big_now or close_now) and centered_req_ok and sharp >= SHARP_MIN_VLM:
                goal_hits = min(GOAL_CONFIRM_N, goal_hits + 1)
            else:
                goal_hits = max(0, goal_hits - 1)
        else:
            goal_hits = max(0, goal_hits - 1)

        if goal_hits >= GOAL_CONFIRM_N and seen:
            at_goal_now = True

        # mode transitions
        if mode == "SEARCH":
            if seen:
                mode = "TRACK"
                rot_streak = 0
                search_waiting = False
                speak_once("track", "Цель вижу. Иду к мячу.")
        else:
            if not seen and (now() - last_seen_ts) > (LOSS_HOLD_SEC + 0.25):
                mode = "SEARCH"
                rot_streak = 0
                search_waiting = False
                speak_once("lost", "Потерял мяч. Ищу.")

        if (not at_goal_prev) and at_goal_now:
            speak_once("goal", "Цель достигнута.", cooldown=2.0)

        # -----------------
        # Decide action (NO BACK)
        # -----------------
        desired = "stop"
        why = ""

        if auto:
            if mode == "TRACK" and sharp < SHARP_MIN_TRACK:
                desired = "stop"
                why = "blur_stop_track"
                if not motion.busy():
                    motion.start_pulse("stop", 0.00, 0.10)

            elif mode == "TRACK":
                if at_goal_now:
                    desired = "stop"
                    why = "goal_hold"
                    if not motion.busy():
                        motion.start_pulse("stop", 0.00, 0.25)
                else:
                    if not seen:
                        mode = "SEARCH"
                        search_waiting = False
                        desired = "stop"
                        why = "track_lost_to_search"
                        if not motion.busy():
                            motion.start_pulse("stop", 0.00, 0.10)
                    else:
                        err = cx_smooth - 0.5
                        fwd_sec = TRACK_FWD_SEC_NEAR if use_area >= 0.18 else TRACK_FWD_SEC_FAR

                        if abs(err) > DEADZONE_CX:
                            rot_sec = TRACK_ROT_SEC_COARSE if abs(err) > (DEADZONE_CX * 2.0) else TRACK_ROT_SEC_FINE
                            desired = rot_left_cmd if err < 0 else rot_right_cmd
                            why = f"track_center err={err:+.2f}"
                            if not motion.busy():
                                motion.start_pulse(desired, rot_sec, TRACK_STAB_SEC)
                                rot_streak += 1
                        else:
                            desired = "fwd"
                            why = "track_fwd"
                            if not motion.busy():
                                motion.start_pulse("fwd", fwd_sec, TRACK_STAB_SEC)
                                rot_streak = 0

                        if rot_streak >= MAX_ROT_STREAK and (not motion.busy()):
                            motion.start_pulse("fwd", 0.10, 0.12)
                            rot_streak = 0

            elif mode == "SEARCH":
                # В SEARCH движение уже делается в блоке SEARCH state-machine.
                desired = "stop" if search_waiting else motion.last_pulse_cmd
                why = "search_wait" if search_waiting else "search_rotate"

        # Draw overlays
        if cv_obs.cand and cv_obs.bbox is not None:
            draw_bbox(frame, cv_obs.bbox, color=(0,165,255), thickness=2)
        if cv_loose.cand and cv_loose.bbox is not None:
            draw_bbox(frame, cv_loose.bbox, color=(0,255,255), thickness=1)

        cv2.circle(frame, (W//2, H//2), 6, (0,165,255), 2)

        lines = []
        lines.append(
            f"v8.4 SEARCH_REACT_FAST_NO_BACK fps={grabber.fps():.1f}/{fps_local:.1f} "
            f"frameAge={(now()-frame_ts):.2f}s sharp={sharp:.1f} auto={'ON' if auto else 'OFF'} "
            f"CV={'ON' if cv_on else 'OFF'} flipX={'ON' if flip_x else 'OFF'} voice={'ON' if voice.enabled else 'OFF'} "
            f"busy={motion.busy()} oneDirScan={one_dir_scan}"
        )
        lines.append(f"MODE={mode} sweep_dir={'L' if sweep_dir<0 else 'R'} step={sweep_step}/{SEARCH_SWEEP_STEPS} rotStreak={rot_streak}")
        lines.append(f"STOP: settled={(now()-motion.last_motion_end_ts):.2f}s still={still} searchWait={search_waiting} wait={(now()-search_wait_t0 if search_waiting else 0.0):.2f}s")
        lines.append(f"CVloose: cand={cv_loose.cand} area={cv_loose.area_frac:.3f} cx={cv_loose.cx:.2f}")
        lines.append(f"CVstrict: cand={cv_obs.cand} confWin={cv_sum}/{CV_CONFIRM_WIN} confirmed={cv_confirmed} suppress={suppress_cv} score={cv_obs.score:.2f} cx={cv_obs.cx:.2f} area={cv_obs.area_frac:.3f}")
        lines.append(f"VLM: enabled={vlm_enabled} ok={vst.ok} busy={vst.busy} age={v_age:.1f}s err={vst.err} goodAge={good_age:.1f}s")
        if good_raw:
            raw1 = good_raw.replace('\\n', ' ')
            lines.append(f"VLMgood: {raw1[:90]}")
        lines.append(f"USED: seen={seen} src={src} cx_s={cx_smooth:.2f} area={use_area:.3f} centered={centered_s} goalHits={goal_hits}/{GOAL_CONFIRM_N} atGoal={at_goal_now}")
        lines.append(f"CMD: {desired}  why={why}")
        lines.append("Keys: Space(AUTO) b(CV) v(VLM now) t(voice) m(flipX) o(oneDirScan) +/- font  q/Esc quit")

        overlay_lines(frame, lines, font_scale=font_scale)
        cv2.imshow(WINDOW_NAME, frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == ord(' '):
            auto = not auto
            if not auto:
                motion.cancel()
                search_waiting = False
        elif k == ord('b'):
            cv_on = not cv_on
            cv_hist = [0] * CV_CONFIRM_WIN
            cv_hist_i = 0
            cv_confirmed = False
            cv_sum = 0
        elif k == ord('v'):
            if vlm_enabled and (not motion.busy()) and (not vlm.get_state().busy) and sharp >= SHARP_MIN_VLM and frame_recent(frame_ts):
                last_analyze_ts = now()
                last_scan_vlm_ts = now()
                vlm.ask_sync(frame, timeout=VLM_TIMEOUT_SEC)
        elif k == ord('t'):
            voice.enabled = not voice.enabled
        elif k == ord('m'):
            flip_x = not flip_x
        elif k == ord('o'):
            one_dir_scan = not one_dir_scan
        elif k in (ord('+'), ord('=')):
            font_scale = min(1.3, font_scale + 0.05)
        elif k in (ord('-'), ord('_')):
            font_scale = max(0.35, font_scale - 0.05)

    motion.cancel()
    grabber.stop()
    voice.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()