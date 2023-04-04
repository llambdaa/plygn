# plygn

| :warning: | This is a research project for my bachelor's degree. Contributions from others are not accepted for now. The project is formally ongoing and I will later need to assure I have completed the work myself. After finishing my studies, the project will be open for collaboration. |
|-----------|:----------|

## ⓘ About
Initially, this project was intended for generation of so-called [Low Poly Art](https://www.google.com/search?q=low+poly+art) from a given image - hence the name **plygn**. However, as that functionality was implemented, it became apparent that by approximating an image using triangles, data redundancy is
generated, because all pixels in a triangle have the same color. That redundancy could then be picked up by various image compression
algorithms (or formats) to compress the image even more.

<!-- Images -->
<!-- Benchmark results -->
<!-- Works even on small images, but less good -->
<!-- Works for every compression algorithm, because they leverage data redundancy -->

## ⚙️ Installation (Linux)
Clone this repository and install multiple dependencies:
```
git clone git@github.com:llambdaa/plygn.git && cd plygn
```
```
sudo apt-get install python3.10 python3-pip
sudo pip3 install argparse opencv-python rawpy numpy numba faiss plotly qoi
```
The first execution will be slow. That is, because [numba](https://github.com/numba/numba) compiles some parts of the script
and caches them on your machine. After that, execution will be considerably faster than before.

## 📒 Usage
| Short | Long | Choices | Default | Description |
| ----- | ---- | ------- | ------- | ----------- |
| -i | --input | - | - | Path to input image |
| -o | --output | - | - | Path to output image |
| -c | --colorspace | RGB, HSL, HSV | RGB | Color space for clustering image data |
| -d | --distance | - | 10 | Preferred vertex distance |
| -s | --splitting | - | -1 | Maximum triangle area before splitting into smaller triangles |
| -v | --variance | - | 1 | Maximum allowed color variance for a triangle to be drawn | 
| -n | --noise-kernel | - | 5 | Kernel size for noise reduction on contours | 
| -k | --kmeans | - | 8 | Centroid count for kmeans color clustering |
| -f | --formats | JPG, PNG, QOI | JPG | Export formats | 
| -B | --benchmark | - | - | Flag for printing and logging compression benchmarks |
| -P | --plot | - | - | Flag for plotting image in selected color space |
| -C | --export-contours | - | - | Flag for exporting images of contours |
| -T | --export-triangulation | - | - | Flag for exporting triangulation of image |
| -U | --export-unprocessed | - | - | Flag for exporting unprocessed image in specified formats for comparison |

© llambdaa / Lukas Rapp 2022-23
