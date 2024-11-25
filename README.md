# WildSemiFusion
A version of WildFusion that utilizes experts to decrease training time with large modularity and improve results



## Rellis Dataset
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


## Training

## Testing



## Not sorted
You must apt-get install libglvnd-dev