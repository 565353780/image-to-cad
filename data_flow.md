# Data Flow

## Input

inputs.batched_inputs

## Data

### ROCA

#### backbone

```bash
inputs.batched_inputs -> inputs.images -> predictions.features
```

#### proposals

```bash
inputs.images -> predictions.proposals
predictions.features -> predictions.proposals
inputs.batched_inputs -> gt_instances[train] -> predictions.proposals
```

### ROCAROI

#### box

```bash
predictions.proposals -> predictions.instances
gt_instances[train] -> predictions.instances
predictions.features[infer] -> predictions.instances
inputs.batched_inputs -> intrinsics
```

#### DepthHead

```bash
inputs.batched_inputs -> inputs.image_depths[train]
predictions.features -> predictions.depths
predictions.features -> predictions.depth_features
```

#### mask

```bash
predictions.instances -> predictions.alignment_instances -> predictions.pool_boxes -> predictions.alignment_features
predictions.instances -> predictions.alignment_instances -> predictions.pool_boxes -> xy_grid, xy_grid_n
predictions.instances -> predictions.alignment_instances -> gt_classes
predictions.features -> predictions.alignment_features -> xy_grid, xy_grid_n
```

