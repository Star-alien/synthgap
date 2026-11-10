import subprocess, os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
def sh(c):
    print("$", c, flush=True); r = subprocess.run(c, shell=True); print("exit", r.returncode, flush=True); return r.returncode
sh("rm -rf /kaggle/working/synthgap; git clone -q https://github.com/Star-alien/synthgap /kaggle/working/synthgap")
os.chdir("/kaggle/working/synthgap")
sh("pip -q install -U diffusers transformers accelerate 2>&1 | tail -1; pip -q install open_clip_torch pytorch-fid; pip -q install -e . 2>&1 | tail -1")
ov = "--set paths.raw=/kaggle/input/datasets/wenewone/cub2002011 --set paths.pools=/kaggle/working/pools --set paths.indices=/kaggle/working/indices --set paths.results=/kaggle/working/results --set paths.features=/kaggle/working/features --set paths.captions=/kaggle/working/captions"
rc = sh(f"python -m synthgap.cli generate --config configs/exp/main.yaml --config configs/exp/kaggle20.yaml {ov} --set generation.families=[sdxl] --set prompts.regimes=[P2] --set generation.batch_size=1 --set generation.resolution=768")
sh("du -sh /kaggle/working/pools; find /kaggle/working/pools -name manifest.json")
sh("rm -rf /kaggle/working/synthgap")
sys.exit(rc)
