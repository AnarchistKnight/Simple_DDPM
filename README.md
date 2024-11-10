The program is to train a simple unconditional DDPM. I was just curious why diffusion model works well and which components really contribute to its working. So I decided to start training a simple DDPM from scratch and conduct some ablation studies.

### Content
- [Dataset](#Dataset)
- [Some Details](#Some-Details)
- [Checkpoint](#Checkpoint)
- [Setup](#Setup)
- [Loss Curve](#Loss-Curve)
- [Generated Images](#Generated-Images)

### Dataset
I used playground v2.5 with prompt "anime-style avatar, totally white background", to generate train and test images. The benefits of doing this is that, the generated images are of a single kind, and I expect this would make the training easier. The GPU at hand is a 4070ti-super with 16 GB memory, so I downsampled the generated images from 1024x1024 to 32x32. As a result of this, I could set batch size to 230. The batch size is large enough to use batch normalizaiton. It's better to use group normalization when the batch size is small, as many papers suggest.

To generate your own data, run
~~~
bash images/generate_images_for_train.sh
bash images/downsample_images_for_train.sh
~~~
and
~~~
bash images/generate_images_for_test.sh
bash images/downsample_images_for_test.sh
~~~

### Some Details
The input is normalized to [-1, 1], some papers claims that input normalization to [0, 1] is also fine. The denoiser is a simple UNet, no attention layer is added. The tail_block at the end of UNet is to map the intermediate output during denoising process to an unbounded range. Well, it seems not very necessary as many implementations choose to clamp the intermediate output to [-1, 1]. I use SiLU as the activation function, as many people do.

### Checkpoint
checkpoint could be downloaded here
~~~
https://drive.google.com/file/d/1-k_7pffTLUT5lSXFNqH9Fml5KPx7Gbos/view?usp=drive_link
~~~

### Setup
My python version is 3.12.4, my cuda version is 12.6. To setup the python environment, run the following scripts in the terminal.
~~~
python -m venv venv
source venv/bin/activate
bash install.sh # to install necessary python packages
~~~

For your convenience, I also paste the full pip list in [pip_list.txt](https://github.com/AnarchistKnight/Simple_DDPM/blob/master/pip_list.txt).

### Train
To train the model, two terminals need to be opened. In one terminal, run
~~~
visdom
~~~
In the other terminal, run
~~~
bash train_UNet.sh
~~~
To see the loss curve, open the following link in your web browser
~~~
http://localhost:8097/
~~~

### Generate Images
To generate images, simply run
~~~
bash generate.sh
~~~

### Loss Curve
Loss curve when trained with 23,000 images
  ![loss curve](https://github.com/AnarchistKnight/Simple_DDPM/blob/master/loss_curve_1.png)
Loss curve when trained with 69,000 images
  ![loss curve](https://github.com/AnarchistKnight/Simple_DDPM/blob/master/loss_curve_2.png)

### Example Generated Images
![generated images](https://github.com/AnarchistKnight/Simple_DDPM/blob/master/generated/generated_images_13x13.png)
