
"""
video_compression.py (solution.py)
Sentio Mind · Project 2 · Smart Behavioral Video Compression
"""

import cv2
import json
import base64
import subprocess
import time
import os
import numpy as np
from pathlib import Path
from PIL import Image
import imagehash

# ---------------------------------------------------------------------------
# CONFIG (Strictly matching the assignment rules)
# ---------------------------------------------------------------------------
VIDEO_IN               = Path(r"C:\\Users\\vinee\\Downloads\\Class_8_cctv_video_1.mov") 
VIDEO_OUT              = Path("compressed_output_1.mp4")
REPORT_HTML_OUT        = Path("compression_report_1.html")
SEGMENTS_JSON_OUT      = Path("segments_kept_1.json")

PHASH_THRESHOLD        = 0.95   # Step 1: drop frame if > 95% similar
MOTION_KEEP_THRESH     = 0.15   # Step 4: Keep if > 0.15
MOTION_DISCARD_THRESH  = 0.05   # Step 2: Discard if < 0.05
CONTEXT_EVERY_SEC      = 3      # Step 5: Keep 1 frame every 3 seconds
OUTPUT_FPS             = 12     
OUTPUT_CRF             = 28     


# ---------------------------------------------------------------------------
# PERCEPTUAL HASH
# ---------------------------------------------------------------------------

def compute_phash(frame: np.ndarray) -> str:
    """Compute a perceptual hash using Extreme Downsampling for speed."""
    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_NEAREST)
    rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    hash_obj = imagehash.phash(pil_img)
    return ''.join(['1' if b else '0' for b in hash_obj.hash.flatten()])

def phash_similarity(h1: str, h2: str) -> float:
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return 1.0 - (hamming_distance / len(h1))


# ---------------------------------------------------------------------------
# MOTION SCORE
# ---------------------------------------------------------------------------

def compute_motion_score(prev_gray, curr_gray: np.ndarray) -> float:
    """Extreme Downscale (80x60) + Lower Iterations makes flow lightning fast."""
    if prev_gray is None:
        return 0.0
        
    prev_tiny = cv2.resize(prev_gray, (80, 60), interpolation=cv2.INTER_NEAREST)
    curr_tiny = cv2.resize(curr_gray, (80, 60), interpolation=cv2.INTER_NEAREST)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_tiny, curr_tiny, None, 
        pyr_scale=0.5, levels=2, winsize=11, 
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


# ---------------------------------------------------------------------------
# FACE PRESENCE CHECK
# ---------------------------------------------------------------------------

def has_face(frame: np.ndarray, cascade) -> bool:
    """Shrunk further + high scaleFactor to double processing speed."""
    small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    eq_gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(
        eq_gray, scaleFactor=1.3, minNeighbors=3, minSize=(10, 10)
    )
    return len(faces) > 0


# ---------------------------------------------------------------------------
# FRAME KEEP DECISION
# ---------------------------------------------------------------------------

def should_keep_frame(frame: np.ndarray,
                      prev_frame,
                      prev_kept_hash: str,
                      last_kept_time_sec: float,
                      current_time_sec: float,
                      cascade) -> tuple:
    
    # 1. Context frames skip all heavy math instantly
    if (current_time_sec - last_kept_time_sec) >= CONTEXT_EVERY_SEC:
        return True, "context_frame", 0.0, False

    # 2. SPEED HACK: Discard instantly if duplicate. Skips all optical/face math.
    current_hash = compute_phash(frame)
    if prev_kept_hash and phash_similarity(current_hash, prev_kept_hash) >= PHASH_THRESHOLD:
        return False, "discarded_duplicate", 0.0, False 

    # 3. Convert to grayscale exactly once to save redundant processing
    curr_small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_NEAREST)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    
    if prev_frame is not None:
        prev_small = cv2.resize(prev_frame, (160, 120), interpolation=cv2.INTER_NEAREST)
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = None

    motion_score = compute_motion_score(prev_gray, curr_gray)
    
    # 4. If motion is high enough, keep it instantly. Skip face detection.
    if motion_score >= MOTION_KEEP_THRESH:
        return True, "motion_above_threshold", motion_score, False
        
    # 5. Only run the heavy face detector if the motion is low/medium
    face_found = has_face(frame, cascade)
    
    if face_found:
        if motion_score > MOTION_DISCARD_THRESH:
            return True, "face_and_motion", motion_score, True
        else:
            return True, "face_detected", motion_score, True
            
    return False, "discarded_static", motion_score, False


# ---------------------------------------------------------------------------
# THUMBNAIL HELPER
# ---------------------------------------------------------------------------

def frame_to_b64_thumb(frame: np.ndarray, width: int = 200) -> str:
    h, w = frame.shape[:2]
    nh = int(h * width / w)
    thumb = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# VIDEO WRITING (THE ULTIMATE RAM PIPE OPTIMIZATION)
# ---------------------------------------------------------------------------

def write_frames_to_video(kept_frames: list, output_path: Path,
                          fps: float, frame_size: tuple):
    if len(kept_frames) == 0:
        print("\n❌ ERROR: 0 frames were kept!")
        return
        
    print(f"\n➔ Encoding {len(kept_frames)} frames instantly via Direct RAM Pipe...")
    
    # Skips OpenCV VideoWriter completely to avoid choking the hard drive with a 3K AVI file
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{frame_size[0]}x{frame_size[1]}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-", # Read directly from Python's RAM (stdin)
        "-c:v", "libx264",
        "-preset", "ultrafast", # Maximum encoding speed
        "-crf", str(OUTPUT_CRF),
        str(output_path)
    ]
    
    # Open FFmpeg silently in the background
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Blast the raw bytes straight from RAM to FFmpeg
    for frame in kept_frames:
        proc.stdin.write(frame.tobytes())
        
    proc.stdin.close()
    proc.wait()


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_compression_report(segments: list, stats: dict, output_path: Path):
    grid_html = ""
    for seg in segments:
        b64 = seg.get("thumbnail_b64", "")
        reason = seg.get("reason_kept", "Unknown")
        sec = seg.get("start_sec", 0.0)
        seg_id = seg.get("segment_id", "?")
        
        grid_html += f"""
        <div style="display:inline-block; margin: 10px; padding: 10px; background: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 220px; text-align: center;">
            <img src="data:image/jpeg;base64,{b64}" style="width: 200px; border-radius: 4px;" alt="Frame"/>
            <p style="margin: 8px 0 4px 0; font-size: 14px;"><strong>Seg {seg_id}</strong> @ {sec}s</p>
            <span style="font-size: 12px; padding: 3px 8px; background: #e0f2fe; color: #0369a1; border-radius: 12px;">{reason}</span>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sentio Mind Compression Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f1f5f9; color: #334155; padding: 20px; margin: 0; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #0f172a; border-bottom: 2px solid #cbd5e1; padding-bottom: 10px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .stat-box {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #2563eb; margin-top: 5px; }}
            .stat-label {{ font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
            .highlight {{ color: #16a34a; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Smart Behavioral Video Compression Report</h1>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">File Size Reduction</div>
                    <div class="stat-value highlight">{stats.get('reduction_pct', 0)}%</div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:5px;">{stats.get('original_size_mb', 0):.2f} MB &rarr; {stats.get('compressed_size_mb', 0):.2f} MB</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Duration Comparison</div>
                    <div class="stat-value">{stats.get('compressed_duration_sec', 0):.1f}s</div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:5px;">Originally {stats.get('original_duration_sec', 0):.1f}s</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Frames Kept</div>
                    <div class="stat-value">{stats.get('frames_kept', 0)} / {stats.get('frames_original', 0)}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Processing Time</div>
                    <div class="stat-value">{stats.get('processing_time_sec', 0):.1f}s</div>
                </div>
            </div>

            <h2>Storyboard Segments</h2>
            <div style="background: #f8fafc; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0;">
                {grid_html}
            </div>
        </div>
    </body>
    </html>
    """
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap          = cv2.VideoCapture(str(VIDEO_IN))
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total / fps_in
    orig_mb      = VIDEO_IN.stat().st_size / 1_000_000

    print(f"Input: {VIDEO_IN}  |  {total} frames  |  {duration:.1f}s  |  {orig_mb:.1f} MB")

    kept_frames = []
    segments    = []
    prev_frame  = None
    prev_hash   = ""
    last_kept_t = -999.0
    cur_seg     = None
    disc_dup    = 0
    disc_stat   = 0

    skip_factor = max(1, int(round(fps_in / OUTPUT_FPS)))

    frame_idx = 0
    while True:
        # FAST READ: Skip useless frames instantly
        ret = cap.grab()
        if not ret:
            break
            
        if frame_idx % skip_factor != 0:
            frame_idx += 1
            continue
            
        # SLOW READ: Actually decode the pixels
        ret, frame = cap.retrieve()
            
        if frame_idx % 1000 == 0:
            print(f"Processing frame {frame_idx} / {total}...")
            
        ts = frame_idx / fps_in

        keep, reason, motion, face = should_keep_frame(
            frame, prev_frame, prev_hash, last_kept_t, ts, cascade
        )

        if keep:
            kept_frames.append(frame.copy())
            prev_hash   = compute_phash(frame)
            last_kept_t = ts

            if cur_seg is None or (ts - cur_seg["end_sec"]) > 2.5:
                if cur_seg:
                    segments.append(cur_seg)
                cur_seg = {
                    "segment_id":            len(segments) + 1,
                    "start_sec":             round(ts, 2),
                    "end_sec":               round(ts, 2),
                    "frames_in_segment":     1,
                    "reason_kept":           reason,
                    "face_count_in_segment": 1 if face else 0,
                    "motion_score_avg":      round(motion, 3),
                    "thumbnail_b64":         frame_to_b64_thumb(frame),
                }
            else:
                cur_seg["end_sec"]               = round(ts, 2)
                cur_seg["frames_in_segment"]    += 1
                cur_seg["face_count_in_segment"] += 1 if face else 0
        else:
            if "duplicate" in reason:
                disc_dup  += 1
            else:
                disc_stat += 1

        prev_frame = frame
        frame_idx += 1

    if cur_seg:
        segments.append(cur_seg)
    cap.release()

    write_frames_to_video(kept_frames, VIDEO_OUT, OUTPUT_FPS, (fw, fh))

    comp_mb = VIDEO_OUT.stat().st_size / 1_000_000 if VIDEO_OUT.exists() else 0.0
    t_end   = time.time()

    stats = {
        "source_video":             str(VIDEO_IN),
        "compressed_video":         str(VIDEO_OUT),
        "original_size_mb":         round(orig_mb, 2),
        "compressed_size_mb":       round(comp_mb, 2),
        "reduction_pct":            round((1 - comp_mb / (orig_mb + 1e-9)) * 100, 1),
        "original_duration_sec":    round(duration, 2),
        "compressed_duration_sec":  round(len(kept_frames) / OUTPUT_FPS, 2),
        "original_fps":             round(fps_in, 2),
        "output_fps":               OUTPUT_FPS,
        "frames_original":          total,
        "frames_kept":              len(kept_frames),
        "processing_time_sec":      round(t_end - t_start, 2),
        "segments":                 segments,
        "frames_discarded_reasons": {
            "near_duplicate_phash": disc_dup,
            "low_motion_no_face":   disc_stat,
            "total_discarded":      total - len(kept_frames),
        },
    }

    with open(SEGMENTS_JSON_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    generate_compression_report(segments, stats, REPORT_HTML_OUT)

    print()
    print("=======================================================")
    print(f"  Done in {stats['processing_time_sec']:.2f}s")
    print(f"  Size:     {orig_mb:.1f} MB  →  {comp_mb:.1f} MB  ({stats['reduction_pct']}% smaller)")
    print(f"  Duration: {duration:.1f}s  →  {stats['compressed_duration_sec']:.1f}s")
    print("=======================================================")