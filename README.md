# plygn

<p align="center">
  Triangulation-based image abstraction for compression and artistic image generation.
</p>
  
---
  
This is a research project for my Bachelor's CS degree. Technically... I have to do it. But the topic was
my choice and thus it was done with a lot of love! If you have any suggestions or ideas, feel free to
tell me!

| :warning: | Contributions from others are not accepted for now. The project is formally still running and I later have to assure that I have done the work myself! After my studies, the project will be open for collaboration! |
|-----------|:----------|

## About
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
and caches them on your machine. <br> After that, execution will be considerably faster than before.

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
| -P | --plot | - | - | Flag for plotting image in selected color space |
| -C | --export-contours | - | - | Flag for exporting images of contours |
| -T | --export-triangulation | - | - | Flag for exporting triangulation of image |
| -O | --export-original | - | - | Flag for exporting original image in specified export formats |

© llambdaa / Lukas Rapp 2022-23
