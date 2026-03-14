installed=$(which nsys | grep nsys)
if [ -z "$installed" ]; then
  apt install nsight-systems-2025.5.2
fi
python roofline_nsys_pipeline.py --iters 200 --warmup 50 --dtype fp16