import random

import sys
from dvclive import Live
import numpy as np

with Live(save_dvc_exp=True, exp_name="test_imsages") as live:
    
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    live.log_param("epochs", epochs)
    base_img = np.ones((500, 500), np.uint8)
    
    for epoch in range(epochs):
        live.log_metric("train/cb_2025/accuracy", epoch + random.random())
        live.log_metric("train/cb_2025/loss", epochs - epoch - random.random())
        live.log_metric("val/cb_2020/accuracy",epoch + random.random() )
        live.log_metric("val/cb_2020/loss", epochs - epoch - random.random())
        
        live.log_image(f"numpy/{live.step}.png", base_img * epoch * 10)
        live.next_step()

if __name__ == "__main__":
    print("Done")
