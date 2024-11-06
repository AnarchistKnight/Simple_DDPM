from utils import generate_image_by_prompt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input arguments for generating image using diffusion models")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--dest_image_dir", type=str, default="generated_images")
    parser.add_argument("--prompt", type=str, default="anime-style avatar, totally white background")
    parser.add_argument("--model_name", type=str, default="playgroundai/playground-v2.5-1024px-aesthetic")
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    generate_image_by_prompt(num_samples=args.num_samples,
                             dest_image_dir=args.dest_image_dir,
                             prompt=args.prompt,
                             model_name=args.model_name,
                             variant=args.variant,
                             num_inference_steps=args.num_inference_steps,
                             guidance_scale=args.guidance_scale,
                             device=args.device)