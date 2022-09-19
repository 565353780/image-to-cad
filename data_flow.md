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

#### box

```bash
proposals, inputs.gt_instances[train]/features[infer] -> instances
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

#### DepthHead

```bash
inputs.batched_inputs -> inputs.image_depths[train]
features -> depth_features -> depths
```

#### AlignmentHead

##### encode_shape

```bash
alignment_instances -> alignment_instance_sizes
depth_features -> shape_features -> shape_code
```

##### roi_depth

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

##### scale

```bash
gt_classes[train]/alignment_instances[infer] -> alignment_classes
alignment_classes, shape_code -> scales_pred
alignment_instances -> scales_gt[train]
```

##### trans

```bash
roi_mask_depths, roi_mask_depth_points, shape_code, alignment_classes -> trans_pred
alignment_instances -> trans_gt
```

##### noc

```bash
roi_mask_depth_points[train]/roi_depth_points[infer] -> depth_points
alignment_instances -> rot_gt
depth_points, trans_pred -> proc_trans_depth_points
shape_code, proc_trans_depth_points, scales_pred -> raw_nocs
raw_nocs, mask_pred -> nocs
```

##### proc

```bash
mask_pred -> proc_has_enough
nocs, proc_trans_depth_points, noc_codes, alignment_classes, proc_has_enough, scales_pred, mask_pred, mask_probs -> proc_solve_rot, proc_solve_trs
proc_solve_trs, proc_has_enough -> trans_pred(update)
proc_solve_rot, proc_has_enough -> rot_pred
```

#### RetrievalHead

```bash
alignment_instances -> pos_cads, neg_cads -> retrieval_pos_cads, retrieval_neg_cads
mask_pred, pos_cads[train] -> retrieval_masks
shape_code, pos_cads[train] -> retrieval_shape_code
nocs, pos_cads[train] -> retrieval_noc_points
raw_nocs, mask_probs / rot_pred, roi_mask_depth_points, mask_pred, scales_pred, trans_pred -> retrieval_noc_points
inputs.batched_inputs, alignment_instance_sizes, alignment_classes, mask_pred, retrieval_noc_points, shape_code -> cad_ids, pred_indices
alignment_instances -> retrieval_pos_cads, retrieval_neg_cads
```

