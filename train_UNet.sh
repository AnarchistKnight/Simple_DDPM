python source/train_unet.py \
--model_checkpoint_dir "checkpoints" \
--learning_rate 0.00001 \
--batch_size 230 \
--num_epochs 5000 \
--noise_scale 1.0 \
--validate_every 10 \
--save_every 5 \
--train_dataset_directory "D:/playgroundv25/images/train_images_downsample" \
--validate_dataset_directory "D:/playgroundv25/images/test_images_downsample" \
--min_loss_path "min_loss_UNet.json"




