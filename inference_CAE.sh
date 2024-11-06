python source/inference_CAE.py \
--groundtruth_image_directory "D:/playgroundv25/downsample_images" \
--pred_image_directory "D:/playgroundv25/reconstructed_downsample_images" \
--config_file_path "config_CAE_denoising.yaml" \
--model_checkpoint "model_state_CAE_denoising.pth" \
--device "cuda" \
--compare_input_output 1