# Tracking and Detection Lab

This project benchmarks object detection and multi-object tracking pipelines on MOT-style datasets such as `MOT15`, `KITTI-17`, and `MOT20-01`. It combines detector backends, tracker backends, visualization utilities, and MOT metric evaluation in one repository.

The main runtime entrypoint is [src/app.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/app.py). Runtime configuration is read from [src/settings.yaml](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/settings.yaml).

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
- [src/libs/mmdetection3d/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/libs/mmdetection3d): vendored OpenMMLab MMDetection3D codebase.

## Software requirements

Use one conda environment named `openmmlab` for the whole project. That environment should contain both the OpenMMLab stack and the detector/tracker/runtime packages used by this repository.

### System software

Install these system-level requirements first:

- Linux is the safest target platform for this repository and for OpenMMLab.
- Conda or Miniconda.
- Python `3.10` inside the conda environment.
- PyTorch compatible with your CPU or CUDA runtime.
- `git`
- GCC 5+ for native extension builds.

### OpenMMLab stack

This repository vendors `MMDetection3D` under [src/libs/mmdetection3d/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/libs/mmdetection3d). The compatibility window present in this repo is:

- `mmcv>=2.0.0rc4,<2.2.0`
- `mmdet>=3.0.0,<3.3.0`
- `mmengine>=0.7.1,<1.0.0`
- `openmim`

Those ranges come from [src/libs/mmdetection3d/requirements/mminstall.txt](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/libs/mmdetection3d/requirements/mminstall.txt) and the official OpenMMLab installation flow.

### Python packages required by this project

Install these packages in the same `openmmlab` environment:

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
- `detectron2`

These cover the current runtime code in [src/app.py](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/app.py), the detector modules under [src/detector/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/detector), the tracker modules under [src/tracker/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/tracker), and the evaluation/visualization utilities under [src/util/](/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/util).

Sources:

- https://github.com/open-mmlab/mmdetection3d
- https://mmdetection3d.readthedocs.io/en/dev/getting_started.html
- https://openmim.readthedocs.io/en/latest/installation.html

## Recommended environment setup

Create and activate the conda environment named `openmmlab`:

```bash
conda create --name openmmlab python=3.10 -y
conda activate openmmlab
python -m pip install --upgrade pip
```

Install PyTorch and torchvision first. Choose the command that matches your machine from the official PyTorch installer.

For CPU-only environments, a typical command is:

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

After PyTorch is installed, install the OpenMMLab stack:

```bash
pip install openmim
mim install 'mmengine>=0.7.1,<1.0.0'
mim install 'mmcv>=2.0.0rc4,<2.2.0'
mim install 'mmdet>=3.0.0,<3.3.0'
```

Then install the rest of the project runtime packages into the same `openmmlab` environment:

```bash
pip install numpy filterpy scikit-image lap motmetrics transformers \
  matplotlib pillow pyyaml opencv-python deep-sort-realtime ultralytics
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

After that, run the project from the same `openmmlab` environment.

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
