# CDL
Pytorch Implementation for CDL: Privileged Information Helps Explain Away Noisy Correspondence. 

## Requirements

CDL is implemented on Python 3.9 and Pytorch 1.12.0, other requirements can be found in requirements.txt.

You can quickly install the requirements by the following instruction.

```
pip install -r requirements.txt
```

## Datasets

- MS-COCO: dataset/dataset_mscoco.py
- Recipe1M: dataset/dataset_recipe1m.py
- Conceptual Captions: dataset/dataset_cc.py

## Training

You can train CDL by modifying the config files in config folder.

- MS-COCO: config/config_mscoco.py
- Recipe1M: config/config_recipe.py
- Conceptual Captions: config/config_cc.py

Then, modify the import in main_cdl.py to run on the corresponding dataset.

```
python main_cdl.py
```

