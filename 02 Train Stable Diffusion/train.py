import subprocess

MODEL_NAME = "stabilityai/stable-diffusion-2"
TRAIN_DIR = "../Datasets/dataset"
OUTPUT_DIR = "/mnt/data/stable_diffusion_2_skin_balanced"
N_STEPS = 50000
RESOLUTION = 512
CHECKPOINT_STEPS = 10000
OVERSAMPLE = True

command = f"accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path={MODEL_NAME} \
  --train_data_dir={TRAIN_DIR} \
  --use_ema \
  --resolution={RESOLUTION} --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision='fp16' \
  --max_train_steps={N_STEPS} \
  --learning_rate=1e-05 \
  --checkpointing_steps={CHECKPOINT_STEPS} \
  --max_grad_norm=1 \
  --image_column='image' \
  --caption_column='text' \
  --oversample={OVERSAMPLE} \
  --lr_scheduler='constant' --lr_warmup_steps=0 \
  --output_dir={OUTPUT_DIR}"

subprocess.run(command, shell=True)