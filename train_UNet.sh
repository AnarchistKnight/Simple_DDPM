python source/train_unet.py \
--model_checkpoint_dir "model_state_UNet" \
--learning_rate 0.00001 \
--batch_size 230 \
--num_epochs 5000 \
--noise_scale 1.0 \
--validate_every 50 \
--save_every 5 \
--train_dataset_directory "D:/playgroundv25/downsample_images" \
--validate_dataset_directory "D:/playgroundv25/downsample_test_images" \
--min_loss_path "min_loss_UNet.json"




