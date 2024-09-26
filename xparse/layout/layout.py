import sys
sys.path.append("/app/MuRe/3rdparty/torch/local")
import numpy as np

from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor

from xparse.base import BaseModelMeta
from xparse.utils import DotDict

class LayoutRecognizer(BaseModelMeta):
    """Layout recognition models."""
    
    def __init__(self, model_path, device='cpu', **kwargs):
        self.model_path = model_path
        self.device = device
        super().__init__()

        config_file = kwargs.get('config_file', None)
        self.ignore_catids = kwargs.get('ignore_catids', [])
        layout_args = {
            'config_file': config_file,
            'resume': False,
            'eval_only': False,
            'opts': ['MODEL.WEIGHT', self.model_path]
        }
        if device == 'cuda':
            layout_args['opts'].append('MODEL.DEVICE', 'cuda')
        else:
            layout_args['opts'].append('MODEL.DEVICE', 'cpu')

        self.cfg = self.__setup_cfg(layout_args, device)
        self.mapping = ["title", "plain text", "abandon", 
                        "figure", "figure_caption", "table", 
                        "table_caption", "table_footnote", 
                        "isolate_formula", "formula_caption"]
        MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]) \
            .thing_classes = self.mapping
        self.predictor = DefaultPredictor(self.cfg)

    def __setup_cfg(self, layout_args, device): 
        _C = get_cfg()
        _C.MODEL.VIT = CN()
        # CoaT model name.
        _C.MODEL.VIT.NAME = ""
        # Output features from CoaT backbone.
        _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]
        _C.MODEL.VIT.IMG_SIZE = [224, 224]
        _C.MODEL.VIT.POS_TYPE = "shared_rel"
        _C.MODEL.VIT.DROP_PATH = 0.
        _C.MODEL.VIT.MODEL_KWARGS = "{}"
        _C.SOLVER.OPTIMIZER = "ADAMW"
        _C.SOLVER.BACKBONE_MULTIPLIER = 1.0
        _C.AUG = CN()
        _C.AUG.DETR = False
        _C.MODEL.IMAGE_ONLY = True
        _C.PUBLAYNET_DATA_DIR_TRAIN = ""
        _C.PUBLAYNET_DATA_DIR_TEST = ""
        _C.FOOTNOTE_DATA_DIR_TRAIN = ""
        _C.FOOTNOTE_DATA_DIR_VAL = ""
        _C.SCIHUB_DATA_DIR_TRAIN = ""
        _C.SCIHUB_DATA_DIR_TEST = ""
        _C.JIAOCAI_DATA_DIR_TRAIN = ""
        _C.JIAOCAI_DATA_DIR_TEST = ""
        _C.ICDAR_DATA_DIR_TRAIN = ""
        _C.ICDAR_DATA_DIR_TEST = ""
        _C.M6DOC_DATA_DIR_TEST = ""
        _C.DOCSTRUCTBENCH_DATA_DIR_TEST = ""
        _C.DOCSTRUCTBENCHv2_DATA_DIR_TEST = ""
        _C.CACHE_DIR = ""
        _C.MODEL.CONFIG_PATH = ""
        # effective update steps would be MAX_ITER/GRADIENT_ACCUMULATION_STEPS
        # maybe need to set MAX_ITER *= GRADIENT_ACCUMULATION_STEPS
        _C.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
    
        _C.merge_from_file(args.config_file)
        _C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
        _C.merge_from_list(args.opts)

        # 使用统一的device配置
        _C.MODEL.DEVICE = device
        _C.freeze()
        args = DotDict(layout_args)
        default_setup(_C, args)
        return _C

    def __call__(self, file_path):
        # supported base64/numpy/image_path
        image = self.read_image(file_path)
        outputs = self.predictor(image)
        layout_dets = []
        boxes = outputs["instances"].to("cpu")._fields["pred_boxes"].tensor.tolist()
        labels = outputs["instances"].to("cpu")._fields["pred_classes"].tolist()
        scores = outputs["instances"].to("cpu")._fields["scores"].tolist()
        for bbox_idx in range(len(boxes)):
            if labels[bbox_idx] in self.ignore_catids:
                continue
            layout_dets.append({
                "category_id": labels[bbox_idx],
                "poly": [
                    boxes[bbox_idx][0], boxes[bbox_idx][1],
                    boxes[bbox_idx][2], boxes[bbox_idx][1],
                    boxes[bbox_idx][2], boxes[bbox_idx][3],
                    boxes[bbox_idx][0], boxes[bbox_idx][3],
                ],
                "score": scores[bbox_idx]
            })
        return layout_dets
        
if __name__ == "__main__":
    model_path = "/Users/hurong/Desktop/citic/codespace/xParse/MuRe/docker/models/Layout/model_final.pth"
    file_path = "/Users/hurong/Desktop/citic/codespace/xParse/Combine/chain/1_1.jpg"
    model = LayoutRecognizer(model_path, device="cpu")
    result = model(file_path)
    print(result)

