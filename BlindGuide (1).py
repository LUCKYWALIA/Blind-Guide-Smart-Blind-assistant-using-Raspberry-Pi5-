import cv2
import pytesseract
import os
import time
import re
import numpy as np
from gtts import gTTS
import uuid
import threading
from collections import deque

# ============================================================
# CONFIG  — tweak these if needed
# ============================================================
DROIDCAM_URL      = "http://10.106.40.85:4747/video"
DISPLAY_W, DISPLAY_H = 640, 480
DETECT_EVERY_N    = 3          # run detection pipeline every Nth frame
SCORE_HISTORY     = 6          # temporal smoothing window
SPEAK_COOLDOWN    = 4          # sec before repeating same alert
THRESH_STOP       = 120000     # raise if too many false STOPs
THRESH_SLOW       = 45000      # raise if too many false SLOWs
CACHE_DIR         = "/tmp/bva_cache"

# ============================================================
# VOICE CACHE  — pre-generate all fixed phrases at startup
# so obstacle alerts play instantly with zero network delay
# ============================================================
FIXED_PHRASES = [
    "Walking mode on",
    "Walking mode off",
    "Stop. Obstacle very close.",
    "Stop. Obstacle on your left.",
    "Stop. Obstacle on your right.",
    "Obstacle ahead. Slow down.",
    "Obstacle on your left. Slow down.",
    "Obstacle on your right. Slow down.",
    "Path is clear.",
    "Reading text. Please wait.",
    "No text found.",
    "Goodbye.",
]

os.makedirs(CACHE_DIR, exist_ok=True)
_voice_cache: dict[str, str] = {}   # phrase → mp3 path

def _build_voice_cache():
    print("Building voice cache...", end=" ", flush=True)
    for phrase in FIXED_PHRASES:
        key  = re.sub(r'[^a-z0-9]', '_', phrase.lower())[:60]
        path = os.path.join(CACHE_DIR, f"{key}.mp3")
        _voice_cache[phrase] = path
        if not os.path.exists(path):
            try:
                gTTS(text=phrase, lang='en', slow=False).save(path)
            except Exception as e:
                print(f"\n  Cache warn [{phrase}]: {e}")
    print("OK")

_build_voice_cache()

# ============================================================
# GLOBALS
# ============================================================
_speak_lock        = threading.Lock()
_is_speaking       = False
_last_msg          = ""
_last_msg_time     = 0.0

walking_mode       = False
current_state      = "clear"
score_history      = deque(maxlen=SCORE_HISTORY)
prev_gray          = None
frame_counter      = 0

# Background frame grabber
_frame_lock        = threading.Lock()
_latest_frame      = None

# ============================================================
# SPEAK  — non-blocking, cached for fixed phrases
# ============================================================
def speak(text: str, force: bool = False):
    global _is_speaking, _last_msg, _last_msg_time

    now = time.time()
    with _speak_lock:
        if not force:
            if _is_speaking:
                return
            if text == _last_msg and now - _last_msg_time < SPEAK_COOLDOWN:
                return
        _last_msg      = text
        _last_msg_time = now

    def _run():
        global _is_speaking
        _is_speaking = True
        try:
            cached = _voice_cache.get(text)
            if cached and os.path.exists(cached):
                os.system(f"mpg123 -q '{cached}'")
            else:
                # dynamic text (OCR output)
                tmp = f"/tmp/bva_live_{uuid.uuid4().hex}.mp3"
                gTTS(text=text, lang='en', slow=False).save(tmp)
                os.system(f"mpg123 -q '{tmp}'")
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as e:
            print(f"[VOICE ERR] {e}")
        finally:
            _is_speaking = False

    threading.Thread(target=_run, daemon=True).start()

# ============================================================
# BACKGROUND FRAME GRABBER
# Keeps cap.read() off the display loop → eliminates lag
# ============================================================
def _grabber(cap):
    global _latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _latest_frame = frame
        else:
            time.sleep(0.03)

# ============================================================
# OCR MODE
# Fixed: uses force=True so result is never silently dropped
# ============================================================
def _ocr_worker(frame):
    # 1. Announce (force so it fires even if previous speech running)
    speak("Reading text. Please wait.", force=True)

    # 2. Preprocess for best Tesseract accuracy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=15)

    # Sharpen
    sharp_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, sharp_k)

    # Binarise
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )

    # 3. OCR  — PSM 6 = uniform block of text
    raw  = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
    text = re.sub(r'[^a-zA-Z0-9.,!? \n]', '', raw)
    text = re.sub(r'\s+', ' ', text).strip()

    print(f"[OCR] '{text[:80]}{'...' if len(text)>80 else ''}'")

    # 4. Wait for announcement to finish, then speak result
    time.sleep(1.0)
    if text:
        speak(text[:400], force=True)
    else:
        speak("No text found.", force=True)

# ============================================================
# OBSTACLE DETECTION  — optimised pipeline
# ============================================================
def detect_obstacle(frame):
    global current_state, prev_gray, frame_counter
    frame_counter += 1

    h, w = frame.shape[:2]

    # ROI — near-full frame
    roi_top, roi_left, roi_right = int(h*0.10), int(w*0.02), int(w*0.98)
    roi     = frame[roi_top:h, roi_left:roi_right]
    roi_h, roi_w = roi.shape[:2]

    gray_roi    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    total_score = 0.0
    zone_scores = {"left": 0.0, "center": 0.0, "right": 0.0}

    # ── Only run heavy pipeline every Nth frame ──────────────────────────
    if frame_counter % DETECT_EVERY_N == 0:

        # Fast blur (5×5)
        blur  = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # Edges
        edges = cv2.Canny(blur, 30, 100)
        k3    = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, k3, iterations=2)
        edges = cv2.erode(edges,  k3, iterations=1)

        # ── Sparse optical flow (Lucas-Kanade) — 10× faster than Farneback
        motion_cols = {"left": 0.0, "center": 0.0, "right": 0.0}
        col_l, col_r = roi_w // 3, roi_w * 2 // 3

        if prev_gray is not None and prev_gray.shape == gray_roi.shape:
            pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=40, qualityLevel=0.2,
                minDistance=10, blockSize=5
            )
            if pts is not None:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray_roi, pts, None,
                    winSize=(11, 11), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                for old, new, ok in zip(pts, new_pts, status):
                    if not ok[0]:
                        continue
                    mag = float(np.hypot(new[0][0]-old[0][0], new[0][1]-old[0][1]))
                    if mag < 1.0:
                        continue
                    cx_pt = float(old[0][0])
                    col   = "left" if cx_pt < col_l else ("center" if cx_pt < col_r else "right")
                    motion_cols[col] += mag

        prev_gray = gray_roi.copy()

        # ── Contours with depth + motion weighting ───────────────────────
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        danger_y, warn_y = roi_h * 2 // 3, roi_h * 1 // 3

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            depth_w = 3.0 if cy > danger_y else (2.0 if cy > warn_y else 1.0)
            col     = "left" if cx < col_l else ("center" if cx < col_r else "right")
            mot_w   = 1.0 + min(motion_cols[col] / 50.0, 0.8)

            zone_scores[col] += area * depth_w * mot_w

        total_score = sum(zone_scores.values())
        score_history.append(total_score)

    if not score_history:
        return

    smoothed = sum(score_history) / len(score_history)

    # ── Determine state ──────────────────────────────────────────────────
    if smoothed > THRESH_STOP:
        state = "stop"
    elif smoothed > THRESH_SLOW:
        state = "slow"
    else:
        state = "clear"

    # ── Speak only on change ─────────────────────────────────────────────
    if state != current_state:
        current_state = state
        has_data  = sum(zone_scores.values()) > 0
        dominant  = max(zone_scores, key=zone_scores.get) if has_data else "center"

        if state == "stop":
            msg = {
                "left":   "Stop. Obstacle on your left.",
                "center": "Stop. Obstacle very close.",
                "right":  "Stop. Obstacle on your right.",
            }[dominant]
        elif state == "slow":
            msg = {
                "left":   "Obstacle on your left. Slow down.",
                "center": "Obstacle ahead. Slow down.",
                "right":  "Obstacle on your right. Slow down.",
            }[dominant]
        else:
            msg = "Path is clear."

        speak(msg)
        print(f"[STATE → {state.upper()}]  smooth={smoothed:.0f}  dominant={dominant}")

    # ── Draw overlay ─────────────────────────────────────────────────────
    COLOR = {"stop": (0,0,255), "slow": (0,255,255), "clear": (0,255,0)}
    LABEL = {"stop": "STOP", "slow": "SLOW DOWN", "clear": "CLEAR PATH"}

    cv2.putText(frame, LABEL[state], (50, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR[state], 3)
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, h), (255,200,0), 2)

    # L/C/R mini bars
    bx, by = w - 155, h - 75
    for i, col in enumerate(["left", "center", "right"]):
        bar = int(np.clip(zone_scores[col] / 2000, 0, 120))
        cv2.rectangle(frame, (bx, by+i*22), (bx+bar, by+i*22+15), COLOR[state], -1)
        cv2.putText(frame, col[0].upper(), (bx-16, by+i*22+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)

# ============================================================
# CAMERA INIT
# ============================================================
cap = cv2.VideoCapture(DROIDCAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

threading.Thread(target=_grabber, args=(cap,), daemon=True).start()
print("Waiting for camera...", end=" ", flush=True)
for _ in range(30):
    time.sleep(0.1)
    with _frame_lock:
        if _latest_frame is not None:
            break
print("ready." if _latest_frame is not None else "TIMEOUT — check DroidCam IP/port.")

# ============================================================
# MAIN LOOP
# ============================================================
print("\n============================")
print("  SMART BLIND ASSISTANT v5")
print("============================")
print("  W = Walking Mode ON/OFF")
print("  R = Read Text (OCR)")
print("  Q = Quit")
print("============================\n")

_ocr_running = False   # guard: don't stack OCR threads

while True:

    # Non-blocking frame grab
    with _frame_lock:
        if _latest_frame is None:
            time.sleep(0.02)
            continue
        frame = _latest_frame.copy()

    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

    # ── Walking mode ─────────────────────────────────────────────────────
    if walking_mode:
        detect_obstacle(frame)
        cv2.putText(frame, "WALK MODE ON", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    else:
        cv2.putText(frame, "WALK MODE OFF", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

    cv2.putText(frame, "W=Walk  R=Read  Q=Quit", (10, DISPLAY_H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    cv2.imshow("Smart Blind Assistant v5", frame)

    key = cv2.waitKey(1) & 0xFF

    # ── W: toggle walk mode ───────────────────────────────────────────────
    if key == ord('w'):
        walking_mode = not walking_mode
        score_history.clear()
        prev_gray = None
        speak("Walking mode on" if walking_mode else "Walking mode off", force=True)

    # ── R: OCR ───────────────────────────────────────────────────────────
    elif key == ord('r'):
        if not _ocr_running:
            _ocr_running = True
            snap = frame.copy()
            def _ocr_and_reset(f):
                global _ocr_running
                _ocr_worker(f)
                _ocr_running = False
            threading.Thread(target=_ocr_and_reset, args=(snap,), daemon=True).start()
        else:
            print("[OCR] already running, please wait")

    # ── Q: quit ───────────────────────────────────────────────────────────
    elif key == ord('q'):
        speak("Goodbye.", force=True)
        time.sleep(1.8)
        break

cap.release()
cv2.destroyAllWindows()
