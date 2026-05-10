# Quad Corner Coordinate Regression

```bash
quad_regression/
│
├── configs/
│   ├── finetune.yaml
│   ├── paths.yaml
│   └── pretrain.yaml
│
├── docs/
│   └── instructions.md
│
├── experiments/
│   ├── analysis/
│   │   ├── compare_backbones.py
│   │   ├── compare_runs.py
│   │   ├── inspect_weights.py
│   │   ├── metric_finetune.py
│   │   ├── metric_pretrain.py
│   │   ├── visualize_coords.py
│   │   └── visualize_errors.py
│   ├── debug/
│   │   ├── debug_backbone_freeze.py
│   │   ├── debug_load_weights.py
│   │   ├── debug_loss.py
│   │   ├── debug_model_build.py
│   │   ├── debug_model_output.py
│   │   └── debug_single_batch.py
│   ├── explore/
│   │   ├── check_annotation.py
│   │   ├── check_dataloader.py
│   │   ├── check_dataset.py
│   │   └── check_transforms.py
│   └── prototype/
│       ├── proto_augmentation.py
│       ├── proto_loss_fn.py
│       └── proto_scheduler.py
│
├── notebooks/
│
├── outputs/
│   ├── finetune/
│   │   ├── archive/
│   │   │   ├── finetune-resnet50-img224-run1.pth
│   │   │   └── finetune-resnet50-img224-run2.pth
│   │   ├── finetune-resnet50-img224.log
│   │   ├── finetune-resnet50-img224.pth
│   │   └── finetune-resnet50-img224.yaml
│   └── pretrain/
│       ├── pretrain-resnet50-img224.log
│       ├── pretrain-resnet50-img224.pth
│       └── pretrain-resnet50-img224.yaml
│
├── scripts/
│   ├── run_finetune.py
│   └── run_pretrain.py
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── path_utils.py
│   │   └── utils.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── coords.py
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── model_builder.py
│   │   ├── quad_regressor.py
│   │   └── simple_cnn.py
│   ├── __init__.py
│   └── engine.py
│
├── tests/
│   ├── models/
│   │   ├── test_model_builder.py
│   │   └── test_quad_regressor.py
│   ├── data/
│   │   ├── test_dataset.py
│   │   └── test_transforms.py
│   ├── utils/
│   │   └── test_config.py
│   └── conftest.py
│
├── .gitignore
└── inference.py
```