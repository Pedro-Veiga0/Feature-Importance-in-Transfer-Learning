# Feature Importance in Transfer Learning

This work was developed for a thesis at the **University of Minho**.

## Thesis overview

The thesis studies the interpretability of object-detection models under
transfer learning and unsupervised domain adaptation. It investigates whether
Class Activation Mapping (CAM) explanations remain faithful, spatially
localized, and reliable when a detector is adapted from a labelled source
domain to an unlabelled target domain.

The experiments compare gradient-based and gradient-free CAM methods on
YOLO-based object detectors. They evaluate explanation quality through:

- **Faithfulness**: whether important regions identified by a CAM influence
  the detector's confidence.
- **Localization**: whether explanations overlap the annotated object regions.
- **Sanity checks**: whether explanations change appropriately when model
  parameters or data are changed.

The underlying adaptation experiments use the ConfMix unsupervised domain
adaptation setting for object detection. The original training project and
its datasets, checkpoints, and additional utilities are not included here;
the scripts expect those resources to be available locally and may require
their paths to be adjusted before execution.

## Repository structure

- `CAM Performance/` - generates and compares CAM maps.
- `Faithfulness Evaluation/` - evaluates explanation faithfulness.
- `Localization Evaluation/` - measures explanation overlap and localization
  metrics.
- `Sanity Checks/` - evaluates explanation robustness and sensitivity.

The scripts are research code and generally require Python, PyTorch,
Ultralytics, `pytorch-grad-cam`, OpenCV, NumPy, pandas, and scikit-image.
