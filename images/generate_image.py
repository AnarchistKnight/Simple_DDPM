import argparse, os, torch
from diffusers import DiffusionPipeline, EDMDPMSolverMultistepScheduler


def generate_image_by_prompt(num_samples,
                             image_save_dir,
                             prompt="anime-style avatar, totally white background",
                             model_name="playgroundai/playground-v2.5-1024px-aesthetic",
                             torch_dtype=torch.float16,
                             variant="fp16",
                             num_inference_steps=50,
                             guidance_scale=0.05,
                             device="cuda"):
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, variant=variant).to(device)
    pipe.scheduler = EDMDPMSolverMultistepScheduler()
    os.makedirs(image_save_dir, exist_ok=True)
    start_index = len(os.listdir(image_save_dir))
    for index in range(start_index, start_index + num_samples):
        image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        image_save_path = os.path.join(image_save_dir, f"{index}.png")
        image.save(image_save_path)
        print("image saved at", image_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input arguments for generating image using diffusion models")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--image_save_dir", type=str, default="generated_images")
    parser.add_argument("--prompt", type=str, default="anime-style avatar, totally white background")
    parser.add_argument("--model_name", type=str, default="playgroundai/playground-v2.5-1024px-aesthetic")
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    generate_image_by_prompt(num_samples=args.num_samples,
                             image_save_dir=args.image_save_dir,
                             prompt=args.prompt,
                             model_name=args.model_name,
                             variant=args.variant,
                             num_inference_steps=args.num_inference_steps,
                             guidance_scale=args.guidance_scale,
                             device=args.device)