# Removing eyeglasses through adversarial training
A SimGAN based approach to removing eyeglasses from images.

## Installing requirements
* ```python3 -m pip install -r requirements.txt```

## Usage
```
usage: main.py [-h] [--batch_size BATCH_SIZE] [--steps STEPS] [--infer]
               [--log_dir LOG_DIR] [--model_dir MODEL_DIR]
               [--samples_dir SAMPLES_DIR] [--output_dir OUTPUT_DIR]
               [--no_glasses_dir NO_GLASSES_DIR] [--glasses_dir GLASSES_DIR]

GAN that removes glasses from pictures

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size to be used in training
  --steps STEPS         Number of training steps to be performed
  --infer               Performs inference for all images in the samples
                        directory using the latest stored model
  --log_dir LOG_DIR     Directory where all the log files reside
  --model_dir MODEL_DIR
                        Directory where the trained model is saved
  --samples_dir SAMPLES_DIR
                        Directory of the images used for validation/inference
  --output_dir OUTPUT_DIR
                        Directory where the output of the inference will be
                        stored
  --no_glasses_dir NO_GLASSES_DIR
                        Directory of the images without glasses for training
  --glasses_dir GLASSES_DIR
                        Directory of the images with glasses for training
```
