from collections import defaultdict
from rfdetrv2.runner import Pipeline, load_config
cfg = load_config("/Users/tinvo0908/Downloads/rfdetrv2/rfdetrv2/configs/train_default.yaml", output_dir="./out", pretrain_weights="rfdetrv2_base")
pipe = Pipeline(cfg=cfg)

dets = pipe.predict("image.png", threshold=0.35, save_path="./out/predicted.png")
