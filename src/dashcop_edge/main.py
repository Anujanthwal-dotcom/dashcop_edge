"""Main entrypoint for DashCop Edge Real-Time Violation Detection."""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running directly as a script (python src/dashcop_edge/main.py)
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dashcop_edge.config import EdgeConfig
    from dashcop_edge.capture.stream_reader import StreamReader
    from dashcop_edge.tracker.online_ca_sort import OnlineCASORT
    from dashcop_edge.models.sac_detector import SACDetector
    from dashcop_edge.models.hnh_detector import HNHDetector
    from dashcop_edge.models.anpr_engine import ANPREngine
    from dashcop_edge.violations.state_manager import ViolationStateManager
    from dashcop_edge.violations.ticket_dispatcher import TicketDispatcher
    from dashcop_edge.visualizer import HUDVisualizer
else:
    from .config import EdgeConfig
    from .capture.stream_reader import StreamReader
    from .tracker.online_ca_sort import OnlineCASORT
    from .models.sac_detector import SACDetector
    from .models.hnh_detector import HNHDetector
    from .models.anpr_engine import ANPREngine
    from .violations.state_manager import ViolationStateManager
    from .violations.ticket_dispatcher import TicketDispatcher
    from .visualizer import HUDVisualizer


def validate_onnx_models(cfg: EdgeConfig) -> None:
    """Strictly validates that all required ONNX models exist before starting."""
    required_models = [
        ("SAC Detector (Riders & Motorcycles)", cfg.models.sac_model),
        ("HNH Detector (Helmet Compliance)", cfg.models.hnh_model),
        ("ANPR License Plate Detector", cfg.models.lp_det_model),
        ("ANPR License Plate OCR", cfg.models.lp_ocr_model),
    ]

    missing = []
    for name, path in required_models:
        if not Path(path).is_file():
            missing.append((name, path))

    if missing:
        print("\n" + "!" * 76, file=sys.stderr)
        print("  CRITICAL ERROR: MISSING REQUIRED ONNX MODEL FILES", file=sys.stderr)
        print("!" * 76, file=sys.stderr)
        print("  DashCop Edge requires all trained ONNX models to exist before running.", file=sys.stderr)
        print("  The following model file(s) are missing:\n", file=sys.stderr)
        for name, path in missing:
            print(f"   * [{name}]", file=sys.stderr)
            print(f"     Expected file: {path}\n", file=sys.stderr)
        print(f"  Please place your trained .onnx model files into the '{cfg.models.weights_dir}' folder.", file=sys.stderr)
        print("  Required files:", file=sys.stderr)
        print("    - weights/sac.onnx", file=sys.stderr)
        print("    - weights/hnh.onnx", file=sys.stderr)
        print("    - weights/lp_det.onnx", file=sys.stderr)
        print("    - weights/lp_ocr.onnx", file=sys.stderr)
        print("!" * 76 + "\n", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DashCop Edge: Real-Time Traffic Violation Detection")
    parser.add_argument("--config", type=str, default="configs/edge_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--mode", type=str, choices=["video", "usb"], default=None, help="Input mode: video or usb")
    parser.add_argument("--source", type=str, default=None, help="Video file path or USB device index (e.g. 0)")
    parser.add_argument("--fps", type=int, default=None, help="Target FPS for rate-pacing video files")
    parser.add_argument("--display", action="store_true", default=None, help="Display live OpenCV HUD window")
    parser.add_argument("--headless", action="store_true", help="Run without UI window (for edge background services)")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = infinite)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EdgeConfig.load_from_yaml(args.config)

    # CLI parameter overrides
    if args.mode:
        cfg.source.mode = args.mode
    if args.source:
        cfg.source.path = args.source
    if args.fps:
        cfg.source.target_fps = args.fps
    if args.headless:
        cfg.display.enabled = False
    elif args.display:
        cfg.display.enabled = True

    print("\n" + "=" * 65)
    print("  DASHCOP EDGE: REAL-TIME TRAFFIC VIOLATION DETECTION")
    print("=" * 65)
    print(f" * Ingestion Mode : {cfg.source.mode.upper()} ({cfg.source.path})")
    print(f" * Target FPS     : {cfg.source.target_fps} (Rate-paced single pass)")
    print(f" * Tracker Engine : Fast Online CA-SORT (Kalman + Bipartite, No-Gurobi)")
    print(f" * Models Dir     : {cfg.models.weights_dir}")
    print(f" * Display HUD    : {'ENABLED' if cfg.display.enabled else 'HEADLESS'}")
    print(f" * Output Dir     : {cfg.violations.output_dir}")
    print("=" * 65 + "\n")

    # Strict pre-flight validation: all ONNX models must exist
    validate_onnx_models(cfg)

    # 1. Initialize Components
    stream_reader = StreamReader(
        source=cfg.source.path,
        mode=cfg.source.mode,
        target_fps=cfg.source.target_fps,
        queue_size=cfg.source.queue_size,
    )

    sac_detector = SACDetector(
        model_path=cfg.models.sac_model,
        provider=cfg.models.provider,
        conf_thresh=cfg.models.confidence_threshold,
    )

    tracker = OnlineCASORT(
        max_age=cfg.tracker.max_age,
        min_hits=cfg.tracker.min_hits,
        iou_threshold=cfg.tracker.iou_threshold,
        assoc_iou_threshold=cfg.tracker.assoc_iou_threshold,
    )

    hnh_detector = HNHDetector(
        model_path=cfg.models.hnh_model,
        provider=cfg.models.provider,
        conf_thresh=cfg.models.confidence_threshold,
    )

    anpr_engine = ANPREngine(
        det_path=cfg.models.lp_det_model,
        ocr_path=cfg.models.lp_ocr_model,
        provider=cfg.models.provider,
    )

    state_manager = ViolationStateManager(
        min_consecutive_frames=cfg.violations.min_consecutive_frames,
        buffer_size=cfg.violations.best_shot_buffer_size,
    )

    ticket_dispatcher = TicketDispatcher(
        anpr_engine=anpr_engine,
        output_dir=cfg.violations.output_dir,
    )
    ticket_dispatcher.start()

    visualizer = HUDVisualizer(window_name=cfg.display.window_name)

    # Graceful shutdown handler
    running = True

    def sig_handler(sig, frame):
        nonlocal running
        print("\n[Shutdown] Received interrupt signal. Stopping pipeline...")
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # 2. Main Real-Time Processing Loop
    stream_reader.start()
    total_processed = 0
    t_prev = time.time()
    fps_smooth = float(cfg.source.target_fps)

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        cfg.display.enabled = False

    try:
        while running:
            t_frame_start = time.time()
            ok, frame_id, frame, ts = stream_reader.read(timeout=0.2)
            if not ok or frame is None:
                continue

            # Step 1: Detect riders, motorcycles, and cross masks via SAC
            detections, masks, cross_masks = sac_detector.detect(frame)

            # Step 2: Track objects and associate riders to motorbikes
            active_tracks = tracker.update(detections, masks, cross_masks)

            # Step 3: Evaluate helmet compliance on riders
            rider_tracks = [t for t in active_tracks if t["cls"] == 0]
            motor_tracks = [t for t in active_tracks if t["cls"] == 1]
            hnh_results = hnh_detector.evaluate_riders(frame, rider_tracks)

            # Step 4: Temporal violation state evaluation
            for m in motor_tracks:
                events = state_manager.update_track(
                    frame_id=frame_id,
                    frame=frame,
                    motor_track=m,
                    rider_tracks=rider_tracks,
                    hnh_results=hnh_results,
                    timestamp=ts,
                )
                for ev in events:
                    ticket_dispatcher.dispatch(ev)

            # Clean up out-of-frame tracks from state buffer
            active_ids = [t["id"] for t in active_tracks]
            state_manager.cleanup_dead_tracks(active_ids)

            # Performance telemetry
            total_processed += 1
            t_frame_end = time.time()
            latency_ms = (t_frame_end - t_frame_start) * 1000.0
            dt = t_frame_end - t_prev
            t_prev = t_frame_end
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

            # Step 5: Live HUD visualization
            if cfg.display.enabled and has_cv2:
                hud_frame = visualizer.render(
                    frame=frame,
                    tracks=active_tracks,
                    hnh_results=hnh_results,
                    fps=fps_smooth,
                    latency_ms=latency_ms,
                    frame_id=frame_id,
                    mode=cfg.source.mode,
                )
                cv2.imshow(cfg.display.window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("[UI] Exit key pressed.")
                    break
            else:
                if total_processed % 30 == 0:
                    print(f"[{cfg.source.mode.upper()}] Frame {frame_id:05d} | FPS: {fps_smooth:4.1f} | Latency: {latency_ms:4.1f}ms | Active Tracks: {len(active_tracks)}")

            if args.max_frames > 0 and total_processed >= args.max_frames:
                print(f"[Run] Reached maximum frame limit ({args.max_frames}).")
                break

    finally:
        print("\n[Shutdown] Cleaning up pipeline threads...")
        stream_reader.stop()
        ticket_dispatcher.stop()
        if cfg.display.enabled and has_cv2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print(f"[Done] Processed {total_processed} frames. Tickets issued: {ticket_dispatcher.total_tickets_generated}.")


if __name__ == "__main__":
    main()
