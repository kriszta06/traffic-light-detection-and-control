"""
Traffic-light detector - classical computer vision (no ML).

Usage:

  # Single image -> annotated image:
  python main.py image data/sample-dayClip6/sample-dayClip6/frames/dayClip6--00198.jpg

  # Folder of frames -> annotated frames written to output/:
  python main.py folder data/sample-dayClip6/sample-dayClip6/frames

  # Live playback window (frames played like a video, detections drawn live):
  python main.py play data/sample-dayClip6/sample-dayClip6/frames --fps 20
  python main.py play data/sample-dayClip6/sample-dayClip6 --fps 20   # also overlays GT

  # Evaluate vs LISA CSV:
  python main.py eval data/sample-dayClip6/sample-dayClip6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from src.detector import detect
from src.tracker import Tracker
from src.evaluate import FrameResult, Summary, evaluate_frame, load_annotations
from src.visualize import draw_detections, draw_ground_truth


OUT_DIR = Path("output")
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _list_frames(folder: Path):
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS)


# ---------------------------------------------------------------------------
# image / folder / eval (as before)
# ---------------------------------------------------------------------------

def cmd_image(path: str) -> int:
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return 1
    dets = detect(img)
    print(f"{path}: {len(dets)} detection(s)")
    for d in dets:
        print(f"  {d.label:>5}  bbox=({d.x1},{d.y1},{d.x2},{d.y2})  score={d.score:.3f}")
    _ensure_out_dir()
    out_path = OUT_DIR / (Path(path).stem + "_det.jpg")
    cv2.imwrite(str(out_path), draw_detections(img, dets))
    print(f"Saved {out_path}")
    return 0


def cmd_folder(folder: str, limit=None) -> int:
    p = Path(folder)
    frames = _list_frames(p)
    if limit:
        frames = frames[:limit]
    _ensure_out_dir()
    sub = OUT_DIR / p.name
    sub.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        dets = detect(img)
        total += len(dets)
        cv2.imwrite(str(sub / f.name), draw_detections(img, dets))
    print(f"Processed {len(frames)} frames, {total} detections. Output: {sub}")
    return 0


def cmd_eval(clip_dir: str, iou_threshold=0.3, save_examples=20, limit=None) -> int:
    p = Path(clip_dir)
    csv_path = p / "frameAnnotationsBOX.csv"
    frames_dir = p / "frames"
    if not csv_path.is_file() or not frames_dir.is_dir():
        print(f"Expected {csv_path} and {frames_dir}", file=sys.stderr)
        return 1

    gts_by_frame = load_annotations(str(csv_path))
    frame_files = _list_frames(frames_dir)
    if limit:
        frame_files = frame_files[:limit]

    summary = Summary()
    _ensure_out_dir()
    examples_dir = OUT_DIR / (p.name + "_examples")
    examples_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        dets = detect(img)
        gts = gts_by_frame.get(f.name, [])
        summary.add(evaluate_frame(dets, gts, iou_threshold))
        if saved < save_examples and (dets or gts):
            vis = draw_ground_truth(img, [(g.x1, g.y1, g.x2, g.y2, g.label) for g in gts])
            vis = draw_detections(vis, dets)
            cv2.imwrite(str(examples_dir / f.name), vis)
            saved += 1

    print(f"Clip: {p.name}")
    print(f"Frames evaluated: {len(frame_files)}")
    print(f"IoU threshold: {iou_threshold}")
    print(summary)
    print(f"Sample annotated frames: {examples_dir}")
    return 0


# ---------------------------------------------------------------------------
# Live playback window
# ---------------------------------------------------------------------------

def _resolve_play_paths(path: str):
    """
    Accept either a frames folder, or a clip folder containing
    frames/ + frameAnnotationsBOX.csv (so GT can be overlaid).

    Returns (frames_dir: Path, gts_by_frame: dict or None, clip_label: str)
    """
    p = Path(path)
    if p.is_dir() and p.name == "frames":
        return p, None, p.parent.name
    csv_path = p / "frameAnnotationsBOX.csv"
    frames_dir = p / "frames"
    if frames_dir.is_dir() and csv_path.is_file():
        return frames_dir, load_annotations(str(csv_path)), p.name
    if frames_dir.is_dir():
        return frames_dir, None, p.name
    if p.is_dir():
        return p, None, p.name
    raise FileNotFoundError(f"Cannot find frames in {path}")


def _hud(img, text, y=24):
    """Draw a small heads-up overlay (top-left)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (8, y - th - 6), (8 + tw + 10, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (13, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def cmd_play(path: str, fps: float = 15.0, loop: bool = False,
             show_gt: bool = True, scale: float = 1.0,
             smooth: bool = True) -> int:
    """
    Open a window and play frames in order, drawing detections (and
    ground-truth, if available) in real time.

    Keyboard:
      q / ESC  -> quit
      space    -> pause / resume
      n / .    -> next frame (while paused)
      p / ,    -> previous frame (while paused)
      +/-      -> faster / slower
      g        -> toggle ground-truth boxes
      s        -> save current annotated frame
    """
    frames_dir, gts_by_frame, clip_label = _resolve_play_paths(path)
    frames = _list_frames(frames_dir)
    if not frames:
        print(f"No frames in {frames_dir}", file=sys.stderr)
        return 1

    win = f"traffic-light-detection : {clip_label}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx = 0
    paused = False
    last_t = time.time()
    tracker = Tracker(min_hits=2, max_age=4, match_dist=40) if smooth else None
    delay_ms = max(1, int(1000.0 / max(fps, 1.0)))
    _ensure_out_dir()
    print(f"Playing {len(frames)} frames from {frames_dir}")
    print("Keys: [space] pause  [n/p] step  [+/-] speed  [g] GT  "
          "[s] save  [q/ESC] quit")

    while True:
        f = frames[idx]
        img = cv2.imread(str(f))
        if img is None:
            idx = (idx + 1) % len(frames)
            continue

        t0 = time.time()
        raw_dets = detect(img)
        dets = tracker.update(raw_dets) if tracker is not None else raw_dets
        det_ms = (time.time() - t0) * 1000.0

        vis = img
        if show_gt and gts_by_frame is not None:
            gts = gts_by_frame.get(f.name, [])
            vis = draw_ground_truth(vis, [(g.x1, g.y1, g.x2, g.y2, g.label) for g in gts])
        vis = draw_detections(vis, dets)

        now = time.time()
        inst_fps = 1.0 / max(now - last_t, 1e-6)
        last_t = now

        red_n = sum(1 for d in dets if d.label == "red")
        green_n = sum(1 for d in dets if d.label == "green")
        _hud(vis,
             f"{f.name}  frame {idx+1}/{len(frames)}  "
             f"red={red_n} green={green_n}  "
             f"det={det_ms:.0f}ms  fps={inst_fps:.1f}  "
             f"{'PAUSED' if paused else f'target {fps:.0f}fps'}")

        if scale != 1.0:
            h, w = vis.shape[:2]
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
        cv2.imshow(win, vis)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord('q'), 27):                  # q / ESC
            break
        elif key == ord(' '):                      # pause
            paused = not paused
        elif key in (ord('n'), ord('.')):          # next
            idx = min(idx + 1, len(frames) - 1)
            paused = True
        elif key in (ord('p'), ord(',')):          # prev
            idx = max(idx - 1, 0)
            paused = True
            if tracker is not None:
                tracker.reset()
        elif key in (ord('+'), ord('=')):
            fps = min(fps * 1.5, 120.0)
            delay_ms = max(1, int(1000.0 / fps))
        elif key == ord('-'):
            fps = max(fps / 1.5, 1.0)
            delay_ms = max(1, int(1000.0 / fps))
        elif key == ord('g'):
            show_gt = not show_gt
        elif key == ord('s'):
            out = OUT_DIR / f"snapshot_{f.stem}.jpg"
            cv2.imwrite(str(out), vis)
            print(f"Saved {out}")
        else:
            if not paused:
                idx += 1
                if idx >= len(frames):
                    if loop:
                        idx = 0
                    else:
                        break

        # Window closed by user (X button)?
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("image", help="Run detector on a single image")
    p_img.add_argument("path")

    p_fld = sub.add_parser("folder", help="Run detector on a folder of frames")
    p_fld.add_argument("path")
    p_fld.add_argument("--limit", type=int, default=None)

    p_ply = sub.add_parser("play", help="Play frames in a window with live detections")
    p_ply.add_argument("path",
                       help="Path to a frames/ folder OR a clip folder "
                            "(containing frames/ and frameAnnotationsBOX.csv)")
    p_ply.add_argument("--fps", type=float, default=15.0,
                       help="Target playback FPS (default 15)")
    p_ply.add_argument("--loop", action="store_true",
                       help="Loop back to start after the last frame")
    p_ply.add_argument("--no-gt", action="store_true",
                       help="Don't draw ground-truth boxes even if CSV is found")
    p_ply.add_argument("--scale", type=float, default=1.0,
                       help="Window scale (e.g. 0.75 for smaller, 1.5 for bigger)")
    p_ply.add_argument("--no-smooth", action="store_true",
                       help="Disable temporal smoothing (raw per-frame detections)")

    p_ev = sub.add_parser("eval", help="Evaluate against a LISA clip's CSV")
    p_ev.add_argument("clip_dir")
    p_ev.add_argument("--iou", type=float, default=0.3)
    p_ev.add_argument("--examples", type=int, default=20)
    p_ev.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "image":
        return cmd_image(args.path)
    if args.cmd == "folder":
        return cmd_folder(args.path, args.limit)
    if args.cmd == "play":
        return cmd_play(args.path, args.fps, args.loop,
                        show_gt=not args.no_gt, scale=args.scale,
                        smooth=not args.no_smooth)
    if args.cmd == "eval":
        return cmd_eval(args.clip_dir, args.iou, args.examples, args.limit)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
