python source/generate_image.py \
--num_samples 50000 \
--dest_image_dir "generated_images" \
--prompt "anime-style avatar, totally white background" \
--model_name "playgroundai/playground-v2.5-1024px-aesthetic" \
--variant "fp16" \
--num_inference_steps 50 \
--guidance_scale 0.05 \
--device "cuda"