<h1 align="center">Some applications of Spectral Mesh Processing</h1>

<h2 align="center">Environment</h2>

Clone the repo:
```bash
git clone https://github.com/mejhana/Spectral-Mesh-Processing.git
```

Set up enviornment
```bash
conda env create -f environment.yml
conda activate PoissonEditing
```

<h2 align="center">Basic Setup</h2>

1. Download 2 images and put them in the images directory
2. Align the source image in the desired location on the destination image and fill the rest of the pixels with black such that source and destination images are of same size
3. For the aligned and filled source image, get a ROI using any [image annotating tool](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html). 
4. If you are using VGG annotation tool, export the annotation as a json file (annotations -> export annotations as json), place this file in the project folder. Rename this file as "annotation.json"
5. Run main.py, change "clone" to False, if you want to implement fill and set "mixGrad" to True if you want to use mixture of gradients