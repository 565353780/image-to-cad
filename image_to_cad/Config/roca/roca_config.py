from math import log
from collections import Counter

from image_to_cad.Config.roca.maskrcnn_config import maskrcnn_config
from image_to_cad.Config.roca.constants import IMAGE_SIZE
from image_to_cad.Model.roca import ROCA
from image_to_cad.Model.roi.roi_head import ROIHead

def roca_config(
    train_data='Scan2CAD_train',
    test_data='Scan2CAD_val',
    batch_size=4,
    num_proposals=128,
    num_classes: int = 17,
    max_iter=100000,
    lr=1e-3,
    workers=0,
    eval_period=100,
    eval_step=False,
    output_dir='./output/',
    base_config: str = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
    anchor_clusters: dict = None,
    min_anchor_size: int = 64,
    noc_scale=10000,
    noc_offset=1,
    depth_scale=1000,
    class_freqs: Counter = None,
    steps=[60000], #  steps=[60000, 80000],
    random_flip: bool = False,
    color_jitter: bool = False,
    batch_average=False,
    depth_res: tuple = IMAGE_SIZE,
    min_nocs: int = 4*4,
    per_category_noc=False,
    noc_weights=True,
    per_category_trans=True,
    noc_weight_head=True,
    noc_weight_skip=False,
    noc_rot_init=False,
    seed=2022,
    gclip=10.0,
    augment=True,
    zero_center=False,
    irls_iters=1,
    confidence_thresh_test=0.5,
):
    cfg = maskrcnn_config(
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        num_proposals=num_proposals,
        num_classes=num_classes,
        max_iter=max_iter,
        lr=lr,
        num_workers=0,
        eval_period=eval_period,
        output_dir=output_dir,
        base_config=base_config,
        custom_mask=True,
        disable_flip=True,
        enable_crop=False,
        anchor_clusters=anchor_clusters,
        min_anchor_size=min_anchor_size
    )

    # Disable resizing of any kind
    cfg.INPUT.MIN_SIZE_TRAIN = min(IMAGE_SIZE)
    cfg.INPUT.MIN_SIZE_TEST = min(IMAGE_SIZE)
    cfg.INPUT.MAX_SIZE_TRAIN = max(IMAGE_SIZE)
    cfg.INPUT.MAX_SIZE_TEST = max(IMAGE_SIZE)

    # Store NOC decoding data
    cfg.INPUT.NOC_SCALE = noc_scale
    cfg.INPUT.NOC_OFFSET = noc_offset
    cfg.INPUT.DEPTH_SCALE = depth_scale
    cfg.INPUT.DEPTH_RES = depth_res
    cfg.INPUT.AUGMENT = augment

    #FIXME: can not remove this
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 16

    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.MODEL.ROI_HEADS.NOC_MIN = min_nocs
    cfg.MODEL.ROI_HEADS.PER_CATEGORY_NOC = per_category_noc
    cfg.MODEL.ROI_HEADS.NOC_WEIGHTS = noc_weights
    cfg.MODEL.ROI_HEADS.PER_CATEGORY_TRANS = per_category_trans
    cfg.MODEL.ROI_HEADS.NOC_WEIGHT_HEAD = noc_weight_head
    cfg.MODEL.ROI_HEADS.NOC_WEIGHT_SKIP = noc_weight_skip
    cfg.MODEL.ROI_HEADS.NOC_ROT_INIT = noc_rot_init
    cfg.MODEL.ROI_HEADS.ZERO_CENTER = zero_center
    cfg.MODEL.ROI_HEADS.IRLS_ITERS = irls_iters

    cfg.MODEL.ROI_HEADS.CONFIDENCE_THRESH_TEST = confidence_thresh_test


    # Set depth config
    cfg.MODEL.DEPTH_BATCH_AVERAGE = batch_average

    # Set optimizer configuration
    cfg.SOLVER.STEPS = tuple(steps)
    cfg.SOLVER.WORKERS = workers
    cfg.SOLVER.CHECKPOINT_PERIOD = eval_period
    cfg.SOLVER.EVAL_STEP = eval_step
    
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = gclip > 0
    # cfg.SOLVER.CLI
    cfg.SOLVER.CLIP_VALUE = gclip

    # Set class scales
    if not class_freqs:
        class_scales = []
    else:
        class_scales = sorted((k, 1 / log(v)) for k, v in class_freqs.items())
        # class_scales = sorted((k, 1 / v) for k, v in class_freqs.items())
        ratio = 1 / max(v for k, v in class_scales)
        class_scales = [(k, v * ratio) for k, v in class_scales]
    cfg.MODEL.CLASS_SCALES = class_scales

    # Custom logic for augmentations
    cfg.INPUT.CUSTOM_FLIP = random_flip
    cfg.INPUT.CUSTOM_JITTER = color_jitter

    # Set the seed
    cfg.SEED = seed

    return cfg

