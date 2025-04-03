# Repository Structure
## 📁 /conda\_environment

Contains all necessary information to set up the required conda environment, including:

- YAML files for installing required packages.
- Instructions for setting up the Python environment.

---

## 📁 /resources

Stores essential pre-trained model checkpoints and supporting files:

- **Pre-trained models**: Checkpoints for feature extractors (CLAM, HIPT, GigaPath, etc.).
- **SlideSegmenter Files**: Additional files needed for WSI segmentation.

---

## 📁 /src

Contains all core Python scripts for WSI processing and model training.

---

## 📁 /additional\_helper\_scripts

Includes useful Python scripts that assisted in various tasks:

- **Segmentation Refinement**: Scripts for manual segmentation mask adjustments.
- **Data Table Processing**: Scripts for converting Excel label files (e.g. the one Jacqueline used to store the pathology reports) into CSV splits.

👉 *If continuing this project, consider merging label files (i.e. excel tables) into a single source and writing helper functions for handling them efficiently.*

---

## 🔬 Pipeline Overview

This repository provides a complete workflow for processing WSIs, extracting features, and training classification models.

### 1️⃣ **Segmentation**

Segmentation is performed using Ruben's SlideSegmenter and Otsu thresholding. The results are saved as a vertical stack of three masks:

- **Top**: SlideSegmenter output
- **Middle**: Intersection of SlideSegmenter results
- **Bottom**: Otsu thresholding result

Run segmentation with:

```sh
python main_segmentation.py --input_folder data/images --output_folder <SEGMENTATION_OUTPUT_LOCATION>
```

This generates (besides the masks):

- `processed_slides.txt` (list of processed slides)
- `tiling_process_list.csv` (used for tiling)

👉 *Modify the **`mask_location`** column in the CSV if using custom segmentation masks (e.g. after refining the masks from main_segmentation.py).*

---

### 2️⃣ **Tiling**

Extracts tiles from WSIs for feature extraction.

```sh
python main_tiling.py --process_list <SEGMENTATION_OUTPUT_LOCATION>/tiling_process_list.csv --save_dir <TILING_OUTPUT_LOCATION>
```

- Failed slides are also recorded in `processed_slides_autogen.csv`, all values in the row are 0.
- This CSV serves as input for feature extraction.

---

### 3️⃣ **Feature Extraction**

Extracts features from tiles using a specified model.

```sh
python main_FE.py --process_list <PATH_TO_CSV> --save_dir <PATH_TO_OUTPUT_DIR> --name <FE_MODEL_NAME> --batch_size <BATCH_SIZE>
```

- Takes as input a CSV with columns: `slide_id`, `slide_location`, `coords_location`.
- Supports hierarchical feature extraction for HIPT (though suboptimal for needle biopsies).
- This step is time-intensive; consider running multiple processes in parallel.

---

### 4️⃣ **Feature Aggregation & Training**

Training is performed using either **CLAM** or **GigaPath** models:

```python
# Modify dataset parameters before training

dataset = Generic_MIL_Dataset(
    csv_path=configs.csv_path,
    data_dir=configs.data_dir,
    shuffle=False,
    seed=configs.seed,
    print_info=True,
    label_dict={'WILDTYPE': 0, 'BRCA2': 1},
    patient_strat=True,
    ignore=[])
```

- `csv_path` should point to a CSV with `slide_id` and `label` columns.
- `label_dict` defines the class labels (e.g., 'WILDTYPE' vs. 'BRCA2').

Run training on SLURM with the provided `sbatch_templates`.

---

### 5️⃣ **Training Output**

Training scripts (`main_clam.py` & `main_gigapath.py`) generate:

- **Trained model parameters** (best/latest epochs)
- **Pickle files**: Validation predictions for further analysis
- **TensorBoard logs**: Monitor training loss over time

---

## 📖 References

### 🔗 **Github Repositories**

- [CLAM](https://github.com/mahmoodlab/CLAM/tree/master)
- [HIPT](https://github.com/mahmoodlab/HIPT)
- [GigaPath](https://github.com/prov-gigapath/prov-gigapath/tree/main)

### 📄 **Research Papers**

- [CLAM paper](https://arxiv.org/abs/2004.09666)
- [HIPT paper](https://arxiv.org/abs/2206.02647)
- [GigaPath paper](https://www.nature.com/articles/s41586-024-07441-w)

---

### 🛠️ **Final Notes**

- set the environment variables in the .env file before running the scripts
- Ensure your data is correctly formatted before training.
- Feel free to modify the scripts to fit new datasets or experimental setups.

Happy Researching! 🚀