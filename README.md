<h1 align="center">FRAME</h1>
<h3 align="center">Floor-aligned Representation for Avatar Motion from Egocentric Video</h3>

This repository contains the official implementation of the paper "FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video", accepted at CVPR 2025.

Table of Contents:

- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [CAD Models](#cad-models)

## Installation

```bash
pip install -e .
```

## Dataset

The dataset can be downloaded from [here](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.XARMQA&faces-redirect=true).  
After downloading `frame_v001.tar.xz`, it should be extracted in any folder.  
A sanity check can be performed by looping through the dataset:

```bash
python scripts/loop.py --data /path/to/frame_v001/folder
```

You can take a look at other available options by running:

```bash
python scripts/loop.py --help
```

## Training

🚧

## CAD Models

Instructions on how to print the CAD models can be found [here](https://github.com/abcamiletto/frame-cad).

## Citation

```bibtex
@article{boscolo2025frame,
title = {FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video},
author = {Boscolo Camiletto, Andrea and Wang, Jian and Alvarado, Eduardo and Dabral, Rishabh and Beeler, Thabo and Habermann, Marc and Theobalt, Christian},
year = {2025},
journal={CVPR},
}
```

## License

The code in this repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.  
The license do not cover the dataset, which is released under a different license. Please refer to the [dataset page](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.XARMQA&faces-redirect=true) for more information.  
The license do not cover any third-party code included in this repository, which is released under their respective licenses.
