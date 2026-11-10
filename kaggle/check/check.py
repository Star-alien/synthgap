import subprocess, time, sys, os
def sh(c):
    print("$", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print(r.stdout[-3000:], r.stderr[-1500:], flush=True)
sh("nvidia-smi")
sh("git clone -q https://github.com/Star-alien/synthgap /kaggle/working/synthgap && ls /kaggle/working/synthgap | head")
sh("ls /kaggle/input/ /kaggle/input/* | head -20")
sh("pip -q install diffusers transformers accelerate 2>&1 | tail -2")
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)
from diffusers import StableDiffusionPipeline
t=time.time()
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to("cuda")
print("load s", time.time()-t, flush=True)
pipe.set_progress_bar_config(disable=True)
t=time.time()
imgs = pipe(["a photo of a Laysan Albatross, a type of bird"]*8, num_inference_steps=50, guidance_scale=2.0).images
print("8 images, 50 steps, s:", time.time()-t, flush=True)
