[![Python application](https://github.com/lhahner/tracking-and-detection-lab/actions/workflows/python-app.yml/badge.svg)](https://github.com/lhahner/tracking-and-detection-lab/actions/workflows/python-app.yml) [![Pylint](https://github.com/lhahner/tracking-and-detection-lab/actions/workflows/pylint.yml/badge.svg)](https://github.com/lhahner/tracking-and-detection-lab/actions/workflows/pylint.yml) [![Dependency Graph](https://github.com/lhahner/tracking-and-detection-lab/actions/workflows/dependabot/update-graph/badge.svg)](https://github.com/lhahner/tracking-and-detection-lab/actions/workflows/dependabot/update-graph) [![CodeQL Advanced](https://github.com/lhahner/tracking-and-detection-lab/actions/workflows/codeql.yml/badge.svg)](https://github.com/lhahner/tracking-and-detection-lab/actions/workflows/codeql.yml)
# Tracking and Detection Lab

This project benchmarks object detection and multi-object tracking pipelines on MOT-style datasets such as `MOT15`, `KITTI-17`, and `MOT20-01`. It combines detector backends, tracker backends, visualization utilities, and MOT metric evaluation in one repository.

The main runtime entrypoint is [src/app.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/app.py). Runtime configuration is read from [src/settings.yaml](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/settings.yaml).

 <div style="display: flex; justify-content: center;">
    <img width="1000" src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzJvNmZuZnRnd3h4b2EwOGdsZGdhbzBma3BxcjlhMXlpejFxNHZkZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/P9lYN2gbWDERpiukXN/giphy.gif">
</div>

## What the project does

- Runs 2D object detectors on image sequences and writes detections in MOT format.
- Tracks detections with `SORT` or `DeepSORT`.
- Evaluates tracking output with `motmetrics`.
- Optionally visualizes RGB frames and tracking boxes.
- Stores benchmark summaries and plotting scripts for experiments.

## Repository layout

- [src/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src): application code, detectors, trackers, utilities, tests, settings.
- [data/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/data): datasets and generated benchmark outputs.
- [benchmarks/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/benchmarks): saved benchmark result files.
- [scripts/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/scripts): plotting and helper scripts.
- [docs/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/docs): report material and documentation assets.

## Software requirements

### System software

Install these system-level requirements first:

- Linux is the safest target platform for this repository.
- Conda or Miniconda.
- Python `3.10` inside the conda environment.
- PyTorch compatible with your CPU or CUDA runtime.
- `git`
- GCC 5+ for native extension builds.
- mmdetection3d directory for config files.

### Python packages required by this project

Install these packages in the same project environment:

- `numpy`
- `filterpy`
- `scikit-image`
- `lap`
- `motmetrics`
- `transformers`
- `torch`
- `torchvision`
- `ultralytics`
- `deep-sort-realtime`
- `Pillow`
- `matplotlib`
- `opencv-python`
- `pyyaml`

These cover the current runtime code in [src/app.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/app.py), the detector modules under [src/detector/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/detector), the tracker modules under [src/tracker/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/tracker), and the evaluation/visualization utilities under [src/util/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/util).

`detectron2` is optional and only required for the `frcnn` and `detectron2` detector backends under [src/detector/frcnn/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/detector/frcnn) and [src/detector/maskfrcnn/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/detector/maskfrcnn).

## Recommended environment setup

The repository includes a setup script that creates a fresh conda environment, installs a compatible PyTorch build, pins `numpy<2` for `motmetrics`, keeps `setuptools<81` for `detectron2`, installs the project requirements, and optionally installs `detectron2`.

From the repository root, run one of:

```bash
bash install.sh
bash install.sh --cuda cu121
bash install.sh --cuda cu124
bash install.sh --without-detectron2
```

The script defaults to:

- conda environment name `track-lab`
- Python `3.10`
- PyTorch `2.5.1`
- torchvision `0.20.1`
- CPU-only install unless `--cuda` is passed

If you want to install manually instead of using the script, follow the same order:

1. Create and activate a conda environment with Python `3.10`.
2. Install `pip`, `wheel`, and `setuptools<81`.
3. Install `torch`, `torchvision`, and `torchaudio` from the official PyTorch index for your CPU/CUDA target.
4. Install `numpy<2`.
5. Install `-r requirements.txt`.
6. Install `detectron2` from source with `--no-build-isolation` if you need the `frcnn` or `detectron2` backends.

After installation, activate the environment and run the project from the repository root:

```bash
conda activate track-lab
PYTHONPATH=src python src/app.py
```

Notes:

- `detectron2` may print a `pkg_resources` deprecation warning. With the pinned `setuptools<81`, this is expected and non-fatal.
- The DETR backend downloads model files from Hugging Face on first use. Without `HF_TOKEN`, downloads still work but may be slower or rate-limited.
- Don't try to use the configs of MMDetection3D outside of the project.
  Clone `https://github.com/open-mmlab/mmdetection3d.git` to a local directory where you can find it.

## Dataset layout

The application expects image sequences laid out similarly to MOT datasets. The current settings file points to:

- images: `data/MOT15/train/KITTI-17/img1`
- ground truth: `data/MOT15/train/KITTI-17/gt/gt.txt`

For each sequence, the detector reads frames from an `img1` directory and writes detections to a `det.txt` file under the configured detection output directory.

## Configuration

Edit [src/settings.yaml](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/settings.yaml) before running the project.

Important settings:

- `paths.mot_root`: directory containing the input frames.
- `paths.detection_path`: directory where `det.txt` will be written.
- `paths.output_root`: directory where tracker output files will be written.
- `paths.ground_truth_path`: MOT ground-truth file used for evaluation.
- `paths.models_root`: detector model path, currently used by YOLO.
- `runtime.detector`: one of `yolo`, `detr`, `frcnn`, or `detectron2`.
- `runtime.tracker`: `sort` or `deepsort`.
- `runtime.datatype`: image file extension such as `jpg` or `png`.
- `runtime.display`: enables matplotlib visualization.
- `runtime.benchmark`: enables `motmetrics` evaluation.

The current file uses absolute local paths. If you want the project to be portable, convert them to paths relative to [src/settings.yaml](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/settings.yaml).

## How to run the project

Run the application from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python src/app.py
```

The script also accepts CLI options defined in [src/app.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/app.py), for example:

```bash
PYTHONPATH=src python src/app.py --display --max_age 3 --min_hits 3 --iou_threshold 0.3
```

In practice, most runtime behavior is controlled by [src/settings.yaml](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/settings.yaml), so update the YAML first and then run the command above.

## Typical workflow

1. Prepare or download a MOT-style dataset under [data/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/data).
2. Edit [src/settings.yaml](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/settings.yaml) to point to the image sequence, model path, output path, detector, and tracker.
3. Install the dependencies required by the chosen detector and tracker.
4. Run `PYTHONPATH=src python src/app.py`.
5. Inspect generated files under [src/output/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/output) and benchmark summaries under [data/benchmark/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/data/benchmark).

## Outputs

The application writes:

- detector output: `det.txt` in the configured detection directory
- tracker output: `{sequence}.txt` in the configured output directory
- benchmark summary: timestamped text files in `data/benchmark`

The plotting utilities in [scripts/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/scripts) can be run separately to visualize benchmark trends.

## Known limitations

- The current YOLO test file imports `detector.YOLO.yolo`, but the actual package path is `detector.yolo.yolo`.
- Some settings currently use machine-specific absolute paths.
- The main application duplicates part of the CLI/runtime configuration logic between arguments and YAML settings.
- The vendored OpenMMLab code is present in the repository, but it is not yet wired into the main runtime pipeline.

## Useful scripts

- [scripts/visualize_lidar.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/scripts/visualize_lidar.py): view KITTI-style LiDAR `.bin` files.
- [scripts/plot_kitti17_sort_benchmarks.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/scripts/plot_kitti17_sort_benchmarks.py): generate KITTI-17 SORT benchmark figures.
- [scripts/plot_kitti17_deepsort_benchmarks.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/scripts/plot_kitti17_deepsort_benchmarks.py): generate KITTI-17 DeepSORT benchmark figures.
- [scripts/plot_mot20_01_sort_benchmarks.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/scripts/plot_mot20_01_sort_benchmarks.py): generate MOT20-01 SORT benchmark figures.
- [scripts/plot_mot20_01_deepsort_benchmarks.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/scripts/plot_mot20_01_deepsort_benchmarks.py): generate MOT20-01 DeepSORT benchmark figures.

## Validation status

The recent docstring-related edits were syntax-checked with `compileall`. Full runtime validation still depends on installing the detector-specific packages for the backend you choose.
