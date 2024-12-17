# WildSemiFusion
This project creates an extension of WildFusion [[1]](#1) that combines the original MultiModal 
model with domain experts (color expert, semantic expert). In addition, it has 
converted the 3D model to a 2D model that predicts color and semantics based off of gray and RGB 
images from the Rellis-3D dataset. Note, the codebase began as a fork of the 
WildFusion repository, so many files have similar structures.

### Environment
make sure to create a source on conda environment and install the requirements
```bash
pip install -r requirements.txt
```


## Downloads (Required)
### Data Download

The original and preprocessed data is available at a private Google Drive link and is not in the GitHub directory. To run training or testing, please download the preprocessed data and maintain the same naming convention as 
you place it into the `inputs` directory that is in the base directory of the 
repository. The folder that holds train, val, and testing shoud be called `rellis_2d_preprocessed`.
To perform reprocessing from scratch, download the original dataset and follow the 
instructions in the Preprocessing Data section nad the data name for it should be `rellis_2d`.

- **Preprocessed Dataset**: [Download](https://drive.google.com/drive/folders/1weWpNm6uA6dwqRIPyKsniyjobtZB2Lu9?usp=sharing)  
- **Original Dataset**: *Link not provided*

---

### Model Download

The best models are stored in the `testing_models` directory. There should be subdirectories:
Note the configuration files on local config does not automatically save under 
testing_models, it saves under saved_models, so you would need to manually move a new 
model

- `base`  
- `color_expert`  
- `semantics_expert`  
- `color_simple`  
- `color_linear`  
- `color_mlp`  
- `semantics_color_simple`  
- `semantics_color_linear`  
- `semantics_color_mlp`  

All these directories should contain `best_model.pth`. The `base` directory should also include weights for the Fourier layer and the two CNNs. If these are unavailable locally, the full models can be downloaded below.

- **Models**: [Download](https://drive.google.com/drive/folders/1VYXjC_vNYjgv6AyAHx7qZ7nzgHmDlhO7?usp=drive_link)

---

### Model Resource Environment

Although training and testing were carefully designed to function on both **CUDA** and **CPU** (not MPS), the model was trained on a **G6 GPU AWS instance**, so it may be difficult to train on local hardware, especially due to memory constraints.  

To alleviate issues, there are options in the `local_config.py` file that can help, including batch sizes and file limits.

- **Inference** successfully ran on an **M2 Mac Air CPU** without excessive memory or timing issues.  
- If your **CUDNN** is not compatible with what was set up on AWS (constant with the VM), you may need to manually specify CPU.  
  - This can be done by searching for "CUDA" in the code and replacing the relevant statements.  
  - Currently, the code provides automatic switching between CUDA and CPU, preferring CUDA if available.  

The model uses **PyTorch 2.3**.

---

## Training
### Plotting
The training script will save plots of the training and validation loss and accuracy 
in the specified saved logs directory. You can specify its interval in the config file

### Model Saves
The training script will save the best model based on the validation loss and also 
saves detailed checkpoints based on the interval specified in the config file.

### Main Script
To train all models you can run the following command. The `--config` flag is optional and defaults to `src.local_config`.
Scratch is also optional and will delete  saved models and logs before training begins 
in the script for the specific model if set to true. Thi will not delete the testing_models directory.
The `--color_model_type` flag is optional and defaults to `all`. The options are `simple`, `linear`,
`mlp`, and `all`.
The `--semantics_color_model_type` flag is optional and defaults to `all`. The options are `simple`, `linear`,
`mlp`, and `all`.
```bash
python -m src.main --config src.aws_config --scratch
```
### Base Model
To train the base model, run the following command. The `--config` flag is optional and defaults to `src.local_config`.
```bash
python -m src.training.base --config src.local_config
```

### Color Expert Model
To train the color expert model, run the following command. The `--config` flag is optional and defaults to `src.local_config`.
Similarly, the `--scratch` flag is optional and will delete saved models and logs 
before training begins but will not delete the testing_models directory.
```bash
python -m src.training.color_expert --config src.local_config
```

### Semantics expert model
To train the semantics expert model, run the following command. The `--config` flag is optional and defaults to `src.local_config`.
Similarly, the `--scratch` flag is optional and will delete saved models and logs but will not delete the testing_models directory.
```bash
python -m src.training.semantics_expert --config src.local_config
```

### Color Model
To train the color model, run the following command. The `--config` flag is optional and defaults to `src.local_config`.
The `--model` flag is optional and defaults to `all`. The options are `simple`, `linear`,
`mlp`, and `all`.
```bash
python -m src.training.color --config src.local_config --model <simple|linear|mlp|all>
```

### Semantics Color Model
To train the semantics color model, run the following command. The `--config` flag is optional and defaults to `src.local_config`.
The `--model` flag is optional and defaults to `all`. The options are `simple`, `linear`,
`mlp`, and `all`.
```bash
python -m src.training.semantics_color --config src.local_config --model <simple|linear|mlp|all>
```

## Testing
### Metrics
The following metrics can be calculated for all the complete models over the test 
dataset,
- MSE for color predictions
- MAE for color predictions
- PSNR for color predictions
- Accuracy for semantic predictions
- Precision for semantic predictions
- Recall for semantic predictions
- F1 for semantic predictions
- IoU for semantic predictions

To get color and semantic segmentation metrics per model, run the following commands, 
these commands will look at the `testing_models` directory for the models.

For the base model,
``` 
python -m evaluation.test_base_model --config src.local_config
```
For the Color models which combine the base model and the color expert. Use 
the `--model` flag to specify the combination type as (simple, linear, mlp, all). The 
default is all.
```
python -m evaluation.test_color_model  --config src.local_config --model <simple|linear|mlp>
```
For the SemanticColor Models which combine the base model with color and 
semantic experts. Use the `--model` flag to specify the type as (simple, linear, mlp, all). The 
default is all.
```
python -m evaluation.test_semantic_color_model --model <simple|linear|mlp>
```


## Not sorted
You must apt-get install libglvnd-dev



## References
<a id="1">[1]</a> 
Y. Liu and B. Chen, “Wildfusion: Multimodal implicit
3d reconstructions in the wild,” 2024. [Online]. Available:
https://arxiv.org/abs/2409.19904

---

## Preprocessing Data (Optional)
You should have the following directory structure after following the first two steps

<p align="center">
  <img src="./docs/Rellis_Directory.png" alt="Rellis 2D Directory" style="width: 45%; display: inline-block; vertical-align: top;"/>
  <img src="./docs/Rellis_Preprocessed_Directory.png" alt="Rellis 2D Directory" style="width: 45%; display: inline-block; vertical-align: top;"/>
</p>

### Downloading Rellis Dataset
The ontology and split for Rellis-3D RGB images and annotations are already provided in the 'input/rellis_2d' folder.
Download the following datasets from Rellis-3D and place their contents them in the 'input/rellis_2d' folder. 
### Preprocessing Rellis Dataset
Run the following command from the project directory to preprocess resizing images, synchronizing images to labels,
and splitting the dataset into train, validation, and test sets.
You can optionally set `--input_dir`, `--output_dir`, and `--split_dir` flags. These flags default to `<WildSemiFusion 
Directory>/input/rellis_2d`,
`<WildSemiFusion Directory>/input/rellis_2d_preprocessed'`, and `<WildSemiFusion Directory>/input/rellis_2d/split>` respectively.


```bash
python3 -m src.data.preprocess_rellis_2d
```
### Processing Rellis Dataset
Commands in `src/data/utils` are being used to create LAB and grayscale copies of an RGB image and to generate noise 
masking in the custom dataloader at `src/data/dataloader_rellis_2d.py`. This dataloader will be ran in the main.py, but
parameters are available at config.py.


