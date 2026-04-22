import glob
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
from skimage import io

from util.coordinate_converter import CoordinateConverter
from util.evaluation import Evaluation
from util.settings_loader import SettingsLoader
from util.visualizer import Visualizer

matplotlib.use("TkAgg")

# Tracking systems
from tracker.DeepSORT.deepSort import DeepSort as DeepSortTracker
from tracker.SORT.sort import Sort


def inference(app, settings):
    if app is None:
        raise ValueError("Application is not definied")
    args = app.parse_args()
    phase = args.phase
    path_detection = []

    evaluation_runner = Evaluation(iou_threshold=0.5)
    app.run_detector_by_argument(
        settings.runtime.detector,
        dataset_path=settings.paths.dataset_path,
        detection_path=settings.paths.detection_path,
        model_path=settings.paths.models_root,
    )

    visualizer = Visualizer(app.datatype)
    if settings.runtime.display:
        visualizer.setup_panel()

    pattern = os.path.join(settings.paths.detection_path, "det.txt")
    for seq_dets_fn in glob.glob(pattern):
        if settings.runtime.tracker.lower() == "deepsort":
            mot_tracker = DeepSortTracker(
                max_age=args.max_age,
                min_hits=args.min_hits,
                iou_threshold=args.iou_threshold,
                bgr=False,  # skimage.io.imread returns RGB
            )
        else:
            mot_tracker = Sort(
                max_age=args.max_age,
                min_hits=args.min_hits,
                iou_threshold=args.iou_threshold,
            )

        seq_dets = np.loadtxt(seq_dets_fn, delimiter=",")
        seq = os.path.basename(os.path.dirname(os.path.dirname(seq_dets_fn)))

        sequence_output_path = Path(settings.paths.output_root) / f"{seq}.txt"
        sequence_output_path.parent.mkdir(parents=True, exist_ok=True)
        sequence_ground_truth_path = Path(settings.paths.ground_truth_path)
        ground_truth_by_frame = {}
        mot_accumulator = None
        metrics_history = {"idf1": [], "motp": [], "mota": []}
        should_visualize_metrics = settings.runtime.display and settings.runtime.benchmark

        if should_visualize_metrics and sequence_ground_truth_path.exists():
            allowed_ground_truth_class_ids = (
                {1}
                if evaluation_runner.should_filter_ground_truth_to_pedestrians(seq)
                else None
            )
            ground_truth_by_frame = evaluation_runner.read_mot_file(
                sequence_ground_truth_path,
                filter_ground_truth_by_confidence=True,
                allowed_class_ids=allowed_ground_truth_class_ids,
            )
            mot_accumulator = evaluation_runner.create_mot_accumulator()
        elif should_visualize_metrics:
            print(
                f"Ground truth file not found for live metrics in {seq}: "
                f"{sequence_ground_truth_path}"
            )

        with open(sequence_output_path, "w") as out_file:
            print("Processing %s." % (seq))
            converter = CoordinateConverter()
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1
                dets = converter.convert2DDetectionToBox(seq_dets, frame)
                app.total_frames += 1

                start_time = time.time()
                if settings.runtime.tracker.lower() == "deepsort":
                    frame_path = os.path.join(
                        settings.paths.dataset_path,
                        f"{frame:06d}.{settings.runtime.datatype}",
                    )
                    frame_img = io.imread(frame_path)
                    trackers = mot_tracker.update(dets, frame=frame_img)
                else:
                    trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                app.total_time += cycle_time

                for d in trackers:
                    print(
                        "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                        % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                        file=out_file,
                    )

                if settings.runtime.display:
                    if mot_accumulator is not None:
                        cumulative_metrics = evaluation_runner.compute_cumulative_tracking_metrics(
                            mot_accumulator=mot_accumulator,
                            frame_number=frame,
                            ground_truth_by_frame=ground_truth_by_frame,
                            trackers=trackers,
                            sequence_name=seq,
                        )
                        for metric_name, metric_value in cumulative_metrics.items():
                            metrics_history[metric_name].append(metric_value)

                    visualizer.visualize_tracking_frame(
                        dataset_path=settings.paths.dataset_path,
                        frame=frame,
                        filetype=settings.runtime.datatype,
                        trackers=trackers,
                        colours=app.colours,
                        metrics_history=metrics_history,
                    )

        if settings.runtime.benchmark:
            if sequence_ground_truth_path.exists():
                evaluation_summary = evaluation_runner.evaluate_sequence(
                    ground_truth_file_path=sequence_ground_truth_path,
                    predicted_tracking_file_path=sequence_output_path,
                    sequence_name=seq,
                )
                print(evaluation_summary)
                benchmark_file_path = evaluation_runner.presist_evaluation(
                    evaluation_summary=evaluation_summary,
                    dataset=seq,
                    detector_name=settings.runtime.detector,
                    tracking_name=settings.runtime.tracker,
                )
                print(f"Saved benchmark summary to {benchmark_file_path}")
            else:
                print(
                    f"Ground truth file not found for {seq}: "
                    f"{sequence_ground_truth_path}"
                )

    if app.total_time > 0:
        fps = app.total_frames / app.total_time
        print(
            "Total Tracking took: %.3f seconds for %d frames or %.1f FPS"
            % (app.total_time, app.total_frames, fps)
        )
    else:
        print(
            "Total Tracking took: %.3f seconds for %d frames "
            "(FPS unavailable: no processing time recorded)"
            % (app.total_time, app.total_frames)
        )
