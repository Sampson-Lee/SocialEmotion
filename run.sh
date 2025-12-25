# export TORCH_LIB=~/anaconda3/envs/social_emotion/lib/python3.12/site-packages/torch/lib
# export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# CUDA_VISIBLE_DEVICES=0 python filter_download.py \
#   --input data/youtube_crawler_candidates/20251219_174723z/embarrassment/metadata.jsonl \
#   --out_jsonl results_embarrassment.jsonl \
#   --keep_dir kept_videos/embarrassment \
#   --tmp_dir tmp_downloads/embarrassment \
#   --target_emotions embarrassment \
#   --verbose
# CUDA_VISIBLE_DEVICES=1 python filter_download.py \
#   --input data/youtube_crawler_candidates/20251219_174723z/guilt/metadata.jsonl \
#   --out_jsonl results_guilt.jsonl \
#   --keep_dir kept_videos/guilt \
#   --tmp_dir tmp_downloads/guilt \
#   --target_emotions guilt \
#   --verbose
# CUDA_VISIBLE_DEVICES=2 python filter_download.py \
#   --input data/youtube_crawler_candidates/20251219_174723z/jealousy/metadata.jsonl \
#   --out_jsonl results_jealousy.jsonl \
#   --keep_dir kept_videos/jealousy \
#   --tmp_dir tmp_downloads/jealousy \
#   --target_emotions jealousy \
#   --verbose

CUDA_VISIBLE_DEVICES=0 python filter_download.py  --yt_cookies www.youtube.com_cookies.txt --input data/youtube_crawler_candidates/20251219_174723z/embarrassment/metadata.jsonl --out_jsonl results_embarrassment.jsonl --keep_dir kept_videos/embarrassment --tmp_dir tmp_downloads/embarrassment --target_emotions embarrassment --verbose --yt_dlp_args "--sleep-interval 8 --max-sleep-interval 20 --concurrent-fragments 1 --retries 3 --fragment-retries 3"
