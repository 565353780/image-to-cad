# Data Flow

## Input

inputs.batched_inputs

## Data

### ROCA

#### backbone

```bash
inputs.batched_inputs -> inputs.images -> features
```

#### proposals

```bash
inputs.batched_inputs -> inputs.gt_instances[train]
inputs.images, features, inputs.gt_instances[train] -> proposals
```

### ROCAROI

#### box

```bash
proposals, inputs.gt_instances[train]/features[infer] -> instances
```

#### DepthHead

```bash
inputs.batched_inputs -> inputs.image_depths[train]
features -> depth_features -> depths
```

#### mask

```bash
instances -> alignment_instances

alignment_instances -> pool_boxes
features, pool_boxes -> alignment_features
pool_boxes, alignment_features -> xy_grid, xy_grid_n
alignment_features -> mask_logits -> mask_probs -> mask_pred
alignment_instances -> gt_classes -> class_weights
alignment_instances, xy_grid_n -> mask_gt
```

### AlignmentHead

#### encode_shape

```bash
alignment_instances -> alignment_instance_sizes
depth_features -> shape_features -> shape_code
```

#### roi_depth

```bash
alignment_instance_sizes -> alignment_sizes
alignment_instances, inputs.batched_inputs[infer] -> intrinsics
xy_grid, depths, intrinsics, xy_grid_n, alignment_sizes -> roi_depths, roi_depth_points
xy_grid, inputs.image_depths, intrinsics, xy_grid_n, alignment_sizes -> roi_gt_depths, roi_gt_depth_points
roi_depths, mask_pred -> roi_mask_depths
roi_depth_points, mask_pred -> roi_mask_depth_points
roi_gt_depths, mask_gt -> roi_mask_gt_depths
roi_gt_depth_points, mask_gt -> roi_mask_gt_depth_points
```

#### scale

```bash
gt_classes[train]/alignment_instances[infer] -> alignment_classes
```

