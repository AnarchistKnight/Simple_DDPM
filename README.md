The program is to train a simple unconditional DDPM. I was just curious why diffusion model works well and which components really contributes to its working well. So I decided to start training a simple DDPM from scratch and conduct some ablation studies。

I used playground v2.5 with prompt "anime-style avatar, totally white background", to generate train and test images. The benefits of doing this is that, the generated images are of a single kind, and I expect this would make the training much easier. As the graphic card I have is one 4070ti-super with 16 GB memory, I downsampled the generated 1024x1024 images to 32x32. As a result of this, I could set batch size to 230. The batch size is large enough to use batch normalizaiton. If you train with higher resolution or smaller graphic memory, it's better to use group normalization as many papers suggest.

The input is normalized to [-1, 1]，some papers claims that input normalization to [0, 1] is also fine. I personally prefer inout normalization to [-1, 1], as standard gaussian distribution is central symmetric aournd zero. The denoiser is a simple UNet, no attention layer is added. The tail_block at the end of UNet is to map the pixel value to an unbounded range, instead of [-1, 1]. Considering the input is a gaussian noise, and the denoising is expected to happen slowly and gradually over 1000 steps, not expecting the intermediate output to range between -1 and 1 is reasonable. Well, many implementations choose to clamp the intermediate output to [-1, 1]. More experiments are supposed to be done on this. I use SiLU as the activation function, as many people do.

My python version is 3.12.4, my cuda version is 12.6. To setup the python environment, run the following scripts in the terminal.

~~~
python -m venv venv
source venv/bin/activate
bash install.sh
~~~

For your convenience, I also paste the full pip list in [pip_list.txt](https://github.com/AnarchistKnight/Simple_DDPM/blob/master/pip_list.txt).
