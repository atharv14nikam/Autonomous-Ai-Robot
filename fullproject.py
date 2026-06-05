
import os
import sys
import threading
import time
import queue
import json
import math
import subprocess
from dotenv import load_dotenv

# Audio & ML libs
try:
    import sounddevice as sd
    import numpy as np
    from vosk import Model, KaldiRecognizer
except Exception as e:
    print("Missing audio dependencies:", e)
    print("Install: pip install sounddevice numpy vosk")
    # We'll still let the user run the UI, but voice will error later.

# TTS (piper)
try:
    from piper.voice import PiperVoice
except Exception as e:
    PiperVoice = None
    print("Piper library not available:", e)

# HTTP for OpenRouter
import requests

# Pygame for display
try:
    import pygame
except Exception as e:
    print("Pygame not installed:", e)
    print("Install: pip install pygame")
    raise

# Load environment (.env)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# -------------------------
# Config (edit as needed)
# -------------------------
# Display (5" default)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480

# Vosk (adjust to your path)
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "/home/raspberry/Documents/saras/vosk-models/vosk-model-en-in-0.5")
DEVICE_INDEX = int(os.getenv("AUDIO_DEVICE_INDEX", "1"))
INPUT_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1

# Piper model path
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "/home/raspberry/Documents/saras/piper_models/en_US-hfc_female-medium.onnx")

# Ollama / Gemma model name (local)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-oss-20b")

# Paths
os.environ["PATH"] += ":/usr/bin:/usr/local/bin:/home/raspberry/.local/bin"

# -------------------------
# Shared state between threads
# -------------------------
face_state_lock = threading.Lock()
face_state = {
    "mode": "idle",      # idle | listening | thinking | speaking | error
    "last_text_in": "",  # recognized text
    "last_text_out": "", # reply text
    "speaking_level": 0.0  # float 0..1 for mouth animation
}

# Commands queue for background AI thread
cmd_queue = queue.Queue()

# -------------------------
# Vosk helper
# -------------------------
vosk_model = None
recognizer = None
audio_q = queue.Queue()

def init_vosk():
    global vosk_model, recognizer
    if not os.path.exists(VOSK_MODEL_PATH):
        print("Vosk model not found at:", VOSK_MODEL_PATH)
        return False
    try:
        print("Loading Vosk model from:", VOSK_MODEL_PATH)
        vosk_model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(vosk_model, TARGET_SAMPLE_RATE)
        return True
    except Exception as e:
        print("Vosk init error:", e)
        return False

def audio_callback(indata, frames, time_info, status):
    if status:
        print("[audio status]", status)
    audio_q.put(bytes(indata))

def listen_once(timeout=10):
    """
    Record from device using sounddevice + VOSK until recognizer AcceptWaveform returns text.
    Returns recognized text or empty string on timeout/error.
    """
    if vosk_model is None or recognizer is None:
        print("Vosk not initialized.")
        return ""
    print("🎤 Listening (timeout {}s)...".format(timeout))
    recognizer.Reset()
    start_time = time.time()
    try:
        with sd.RawInputStream(
            samplerate=INPUT_SAMPLE_RATE,
            blocksize=16000,
            device=DEVICE_INDEX,
            dtype="int16",
            channels=CHANNELS,
            callback=audio_callback,
        ):
            while True:
                try:
                    data = audio_q.get(timeout=0.5)
                except queue.Empty:
                    # check timeout
                    if time.time() - start_time > timeout:
                        print("Listen timeout.")
                        return ""
                    continue

                audio_int16 = np.frombuffer(data, dtype=np.int16)
                if INPUT_SAMPLE_RATE != TARGET_SAMPLE_RATE:
                    # Use scipy.resample if available
                    try:
                        import scipy.signal
                        audio_int16 = scipy.signal.resample(
                            audio_int16,
                            int(len(audio_int16) * TARGET_SAMPLE_RATE / INPUT_SAMPLE_RATE)
                        ).astype(np.int16)
                        data = audio_int16.tobytes()
                    except Exception:
                        # fallback: ignore resampling (may break)
                        pass

                if recognizer.AcceptWaveform(data):
                    res = json.loads(recognizer.Result())
                    text = res.get("text", "").strip()
                    print("Vosk result:", text)
                    return text
                # else continue partial
                if time.time() - start_time > timeout:
                    print("Listen timeout.")
                    return ""
    except Exception as e:
        print("listen_once error:", e)
        return ""

# -------------------------
# Piper TTS helper
# -------------------------
voice = None
def init_piper():
    global voice
    if PiperVoice is None:
        print("Piper not available.")
        return False
    if not os.path.exists(PIPER_MODEL_PATH):
        print("Piper model not found:", PIPER_MODEL_PATH)
        return False
    try:
        print("Loading Piper model:", PIPER_MODEL_PATH)
        voice = PiperVoice.load(PIPER_MODEL_PATH)
        print("Piper loaded")
        return True
    except Exception as e:
        print("Piper init error:", e)
        return False

def speak_stream(text):
    """Synthesize text using piper and play to default audio output (blocking)."""
    if voice is None:
        print("Piper voice not initialized.")
        return
    try:
        # We'll generate chunks and stream via sounddevice
        # Use sample rate 44100 and stereo output
        samplerate = 44100
        with sd.OutputStream(samplerate=samplerate, channels=2, dtype="int16", device="default") as stream:
            for chunk in voice.synthesize(text):
                # chunk may be bytes or object with .audio attribute
                if hasattr(chunk, "audio"):
                    audio_data = chunk.audio
                else:
                    audio_data = chunk
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                # If mono, convert to stereo
                if audio_np.ndim == 1:
                    stereo = np.repeat(audio_np[:, None], 2, axis=1)
                else:
                    stereo = audio_np
                stream.write(stereo)
    except Exception as e:
        print("TTS error:", e)

# -------------------------
# Query local Gemma via Ollama (subprocess)
# -------------------------
def query_gemma(prompt, timeout=40):
    try:
        proc = subprocess.run(
            ["/usr/bin/ollama", "run", OLLAMA_MODEL],
            input=prompt.encode(),
            capture_output=True,
            timeout=timeout
        )
        out = proc.stdout.decode().strip()
        if proc.returncode != 0 or not out:
            raise RuntimeError(proc.stderr.decode().strip() or "no output")
        return out
    except Exception as e:
        return f"[Gemma error] {e}"

# -------------------------
# Query OpenRouter cloud
# -------------------------
def query_openrouter(prompt, timeout=60):
    try:
        if not OPENROUTER_API_KEY:
            return "[OpenRouter error] Missing API key in .env"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OpenRouter error] {e}"

# -------------------------
# Hybrid decision logic (used in background thread)
# -------------------------
def ask_robot(prompt):
    """
    If prompt starts with /cloud -> force OpenRouter
    Else try local Gemma first, fallback to OpenRouter if short/error.
    """
    if prompt.startswith("/cloud"):
        cleaned = prompt.replace("/cloud", "").strip()
        return query_openrouter(cleaned)
    local = query_gemma(prompt)
    if not local or "error" in local.lower() or len(local.split()) < 5:
        print("Falling back to cloud...")
        return query_openrouter(prompt)
    return local

# -------------------------
# Background AI Thread
# -------------------------
class HybridAIThread(threading.Thread):
    def __init__(self, cmd_queue, state_dict, state_lock):
        super().__init__(daemon=True)
        self.cmd_queue = cmd_queue
        self.state = state_dict
        self.lock = state_lock
        self.running = True

    def set_state(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def run(self):
        # initialize models here (in background thread)
        vosk_ready = init_vosk()
        piper_ready = init_piper()

        while self.running:
            try:
                cmd = self.cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if cmd == "stop":
                self.running = False
                break
            if cmd == "voice":
                # Start voice flow
                try:
                    self.set_state(mode="listening")
                    text = listen_once(timeout=10)
                    if not text:
                        self.set_state(mode="idle")
                        continue
                    self.set_state(last_text_in=text, mode="thinking")
                    # Query AI
                    reply = ask_robot(text)
                    self.set_state(last_text_out=reply, mode="speaking")
                    # While speaking, periodically update speaking_level for animation
                    # We'll spawn a small thread that updates speaking_level while TTS runs
                    speaking_flag = {"on": True}
                    def tts_and_anim():
                        try:
                            # Synthesize and play; while playing, update speaking_level
                            # Spawn tts in same thread for simplicity
                            # We'll update speaking_level based on an artificial envelope
                            # Start tts in a nested function to allow level updates
                            def tts_play():
                                speak_stream(reply)
                            t_start = time.time()
                            # Start tts (blocking)
                            tts_play()
                        except Exception as e:
                            print("TTS inner error:", e)
                        finally:
                            speaking_flag["on"] = False

                    anim_thread = threading.Thread(target=tts_and_anim, daemon=True)
                    anim_thread.start()
                    # Update speaking_level while anim thread is alive
                    while speaking_flag["on"]:
                        # Simple oscillation for mouth animation
                        t = time.time()
                        lvl = (math.sin(t*6) + 1) / 2  # 0..1
                        self.set_state(speaking_level=lvl)
                        time.sleep(0.05)
                    self.set_state(speaking_level=0.0, mode="idle")
                except Exception as e:
                    print("HybridAIThread error:", e)
                    self.set_state(mode="error")
                    time.sleep(1)
            else:
                # Unknown command
                continue

# -------------------------
# Pygame Face Drawing
# -------------------------
def draw_face(screen, w, h, state, t):
    """
    state: dict with mode, speaking_level, etc.
    t: current time in seconds for animation
    """
    # Colors
    bg = (180, 235, 255)
    black = (10, 10, 10)
    white = (255, 255, 255)
    blush = (255, 150, 170)
    dark_gray = (50, 50, 50)

    screen.fill(bg)

    s = min(w, h)
    eye_r = int(0.12 * s)
    pupil_r = int(0.55 * eye_r)
    shine_r = int(0.22 * eye_r)
    blush_r = int(0.08 * s)

    left_eye = (int(0.28 * w), int(0.36 * h))
    right_eye = (int(0.72 * w), int(0.36 * h))

    mode = state.get("mode", "idle")
    speaking_level = float(state.get("speaking_level", 0.0))

    # Eye animation parameters
    # Blink: when idle blink occasionally
    # When listening, make eyes slightly wider and a small pulsing
    blink = False
    eye_scale = 1.0
    if mode == "listening":
        # open wider and gently pulse
        eye_scale = 1.0 + 0.08 * math.sin(t*6)
    elif mode == "speaking":
        eye_scale = 1.0
    elif mode == "thinking":
        eye_scale = 0.95 + 0.02 * math.sin(t*3)
    elif mode == "error":
        eye_scale = 0.8

    # simple automatic blink timer when idle
    if mode == "idle":
        # Blink every ~3-6 seconds
        if int(t) % 4 == 0 and (t % 1.0) < 0.12:
            blink = True

    # Draw eyes (outer)
    def draw_eye(center, r, scale, is_blink):
        cx, cy = center
        rr = int(r * scale)
        if is_blink:
            # draw a closed eye line
            pygame.draw.ellipse(screen, black, pygame.Rect(cx-rr, cy-int(rr*0.3), rr*2, int(rr*0.6)))
            pygame.draw.line(screen, white, (cx-rr, cy), (cx+rr, cy), 2)
        else:
            pygame.draw.circle(screen, black, center, rr)
            # pupil
            # when listening, pupils dilate slightly
            pr = int(pupil_r * (1.0 + (0.15 if mode == "listening" else 0.0)))
            pygame.draw.circle(screen, white, (cx - int(rr*0.08), cy - int(rr*0.08)), int(pr*0.18))  # highlight small first
            pygame.draw.circle(screen, (30,30,30), center, pr)
            # inner shine
            hx = int(0.35 * rr)
            hy = int(-0.30 * rr)
            shine_pos = (cx + hx, cy + hy)
            pygame.draw.circle(screen, white, shine_pos, int(shine_r * scale))

    draw_eye(left_eye, eye_r, eye_scale, blink)
    draw_eye(right_eye, eye_r, eye_scale, blink)

    # Blush
    left_blush = (int(0.22 * w), int(0.52 * h))
    right_blush = (int(0.78 * w), int(0.52 * h))
    pygame.draw.circle(screen, blush, left_blush, int(blush_r * (1.0 + 0.05 * math.sin(t*3))))
    pygame.draw.circle(screen, blush, right_blush, int(blush_r * (1.0 + 0.05 * math.sin(t*3))))

    # Smile / mouth
    mouth_center = (w // 2, int(0.65 * h))
    mouth_w = int(0.55 * w)
    mouth_h = int(0.28 * h)
    mouth_rect = pygame.Rect(0, 0, mouth_w, mouth_h)
    mouth_rect.center = mouth_center

    # speaking_level 0..1 controls mouth openness
    if mode == "speaking":
        openness = 0.12 + 0.38 * speaking_level  # 0.12..0.5 relative to s
    elif mode == "listening":
        openness = 0.08
    elif mode == "thinking":
        openness = 0.06 + 0.06 * (math.sin(t*4)+1)/2
    elif mode == "error":
        openness = 0.0
    else:
        openness = 0.10

    # Draw mouth as arc / filled ellipse to mimic smile and opening
    # top curve (smile)
    start_ang = math.radians(200)
    end_ang = math.radians(340)
    thickness = max(2, int(0.02 * s))
    # Draw smile line
    pygame.draw.arc(screen, black, mouth_rect, start_ang, end_ang, thickness)

    # Mouth opening: draw an ellipse underneath the smile with height based on openness
    open_h = int(openness * s)
    if open_h > 2:
        mouth_open_rect = pygame.Rect(0,0, int(mouth_w*0.6), open_h)
        mouth_open_rect.center = (mouth_center[0], mouth_center[1] + int(mouth_h*0.18))
        pygame.draw.ellipse(screen, black, mouth_open_rect)
        # inner tongue / highlight when speaking
        inner_rect = mouth_open_rect.inflate(-int(mouth_open_rect.width*0.12), -int(mouth_open_rect.height*0.2))
        pygame.draw.ellipse(screen, (150,30,40), inner_rect)

    # subtle eye rings (semi-transparent)
    ring_surf = pygame.Surface((w,h), pygame.SRCALPHA)
    ring_color = (0,0,0,30)
    pygame.draw.circle(ring_surf, ring_color, left_eye, int(eye_r*1.1), width=6)
    pygame.draw.circle(ring_surf, ring_color, right_eye, int(eye_r*1.1), width=6)
    screen.blit(ring_surf, (0,0))

    # optional: small status dot (top-right) - colored by mode
    status_colors = {
        "idle": (50,200,50),
        "listening": (60,120,255),
        "thinking": (255,200,40),
        "speaking": (20,180,220),
        "error": (255,40,40)
    }
    sc = status_colors.get(mode, (200,200,200))
    pygame.draw.circle(screen, sc, (w-28,28), 12)

# -------------------------
# Main
# -------------------------
def main():
    # init Pygame
    pygame.init()
    pygame.display.set_caption("Hybrid Face Robot - 5\"")
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    clock = pygame.time.Clock()

    # Start background Hybrid AI thread
    ai_thread = HybridAIThread(cmd_queue, face_state, face_state_lock)
    ai_thread.start()

    running = True
    last_t = time.time()

    try:
        while running:
            t = time.time()
            dt = t - last_t
            last_t = t

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                    break
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        break
                    if ev.key == pygame.K_RETURN:
                        # trigger voice command
                        try:
                            cmd_queue.put_nowait("voice")
                        except queue.Full:
                            pass

            # copy state snapshot under lock
            with face_state_lock:
                state_snapshot = dict(face_state)

            draw_face(screen, DISPLAY_WIDTH, DISPLAY_HEIGHT, state_snapshot, t)
            pygame.display.flip()
            clock.tick(30)  # 30 FPS

    except KeyboardInterrupt:
        running = False
    finally:
        # signal background thread to stop
        try:
            cmd_queue.put_nowait("stop")
        except Exception:
            pass
        ai_thread.running = False
        ai_thread.join(timeout=2)
        pygame.quit()
        print("Exited cleanly.")

if __name__ == "__main__":
    main()
