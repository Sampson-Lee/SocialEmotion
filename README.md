Installation
```
# Recommended: Python 3.11/3.12 for best third-party wheel support.
# (Python 3.14 may work for core deps, but some optional CUDA extensions like
# flash-attn often don't publish cp314 wheels and will try to build from source.)
conda create -n social_emotion python=3.12 -y
conda activate social_emotion

pip install -U torch torchvision torchaudio
pip install transformers==4.52.3
pip install accelerate
pip install -U qwen-omni-utils[decord]
pip install -U flash-attn --no-build-isolation

conda install -c conda-forge yt-dlp ffmpeg
conda install -c conda-forge nodejs
conda install -c conda-forge deno
```
