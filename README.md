The program is to train a simple unconditional DDPM. I was just curious why diffusion model works well and which components really contribute to its working. So I decided to start training a simple DDPM from scratch and conduct some ablation studies。

I used playground v2.5 with prompt "anime-style avatar, totally white background", to generate train and test images. The benefits of doing this is that, the generated images are of a single kind, and I expect this would make the training easier. The GPU at hand is a 4070ti-super with 16 GB memory, so I downsampled the generated images from 1024x1024 to 32x32. As a result of this, I could set batch size to 230. The batch size is large enough to use batch normalizaiton. It's better to use group normalization when the batch size is small, as many papers suggest.

The input is normalized to [-1, 1], some papers claims that input normalization to [0, 1] is also fine. The denoiser is a simple UNet, no attention layer is added. The tail_block at the end of UNet is to map the intermediate output during denoising process to an unbounded range. Well, it seems not very necessary as many implementations choose to clamp the intermediate output to [-1, 1]. I use SiLU as the activation function, as many people do.

My python version is 3.12.4, my cuda version is 12.6. To setup the python environment, run the following scripts in the terminal.

~~~
python -m venv venv
source venv/bin/activate
bash install.sh # to install necessary python packages
~~~

For your convenience, I also paste the full pip list in [pip_list.txt](https://github.com/AnarchistKnight/Simple_DDPM/blob/master/pip_list.txt).
