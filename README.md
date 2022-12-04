# plygn

<p align="center">
  Triangulation-based image abstraction for compression and artistic image generation.
</p>
  
---
  
This is a research project for my Bachelor's CS degree. Technically... I have to do it. But the topic was
my choice and thus it was done with a lot of love! If you have any suggestions or ideas, feel free to
tell me!

| :warning: | Contributions from others are not accepted for now. The project is still running formally and I later have to assure that I have done the work myself! After my studies, the project will be open for collaboration! |
|-----------|:----------|

## About
Initially, this project was intended for generation of so-called [Low Poly Art](https://www.google.com/search?q=low+poly+art) from a given image - hence the name **plygn**.
However, as that functionality was implemented, it came to me that by approximating an image using triangles, data redundancy is
generated, because all pixels in a triangle have the same color. That redundancy could then be picked up by various image compression
algorithms (or formats) to compress the image even more.

Of coarse... _**clearing throat**_ I mean, of course, that will work with large triangles. But it won't look anything like the original.
The compression would come with a massive loss in image quality. However, the triangles can be configured to be smaller and smaller,
coming closer to the original instead of showing an obvious abstraction of the image.

< Images Will Follow >

And ideed, it works! I managed to push JPG compression even more. Among the images I tested, the ones that were pre-processed before
being exported as JPG were all between **0% and 60% smaller** than the same image exported as JPG without applying the triangulation technique.

< Exact Inspection Graph Will Follow >

The good part:
  * very interesting topic for my studies
  * general approach (pre-processing can be applied before exporting into any image format)

The not-so-good part:
  * relatively slow (somewhere between 3s and 60s per 4K image - implemented in Python without GPU support)
  * compression rates greatly vary depending on image and degree of details
  * works best for large images (~4K)

Actually, it seems like for PNG, very similar compression rates are achieved. That is because this technique does pre-processing. Details, especially
in large areas of a similar color, are reduced and instead redundancy is created. Any compression algorithm benefits from redundancy. So, it
does not really matter whether the triangulated image is exported as JPG, PNG or QOI (mostly RLE-based encoding). All of them will now
find a lot of information that can be discarded or compacted.

However, triangles have a minimum size. And they do remove details from the area they cover. That information could have been needed
to express depth or edge sharpness. As a result, this technique makes smaller images blurry. But for larger images (4K), this degrading
in sharpness will most likely not be visible, making it look like the original.

## Installation (Linux)
First, clone the repository:
```
git clone git@github.com:llambdaa/plygn.git && cd plygn
```

Before running the script, you must install python3 and multiple dependencies:
```
sudo apt-get install python3.10 python3-pip
sudo pip3 install argparse opencv-python rawpy numpy numba faiss plotly
```

The first time you run the script on an example image, it will be slow.
That is, because [numba](https://github.com/numba/numba) compiles some parts of it and caches them on your machine.
After that, it will run blazingly fast! At least... multiple times faster than before.

## Usage
The algorithm can be configured using program arguments. Some of them work in conjunction with each other
and amplify or damp their effects. You will understand them best if you have read about the inner workings.

 * `-i || --input`: Specifies image input path
 * `-o || --output`: Specifies image output path
 * `-c || --colorspace`: Specifies color space in which the image's pixel colors are plotted
 * `-k || --kmeans`: Specifies amount of dominant colors searched in color space
 * `-d || --distance`: Specifies preferred distance between vertices along contour lines
 * `-n || --noise-kernel`: Specifies matrix size for removing noise in contour lines
 * `-s || --splitting`: Specifies maximum area of triangle before splitting it into smaller triangles
 * `-P || --show-plot`: Toggles plotting image's pixel colors in color space
 * `-C || --show-contour`: Toggles exporting image with contour lines drawn in it
 * `-T || --show-triangulation`: Toggles exporting image with found triangulation drawn in it

Example:
```
./plygn.py -i "~/Downloads/example.png" -o "~/Downloads" -c RGB -d 10 -k 32 -n 0 -s 30
```
**plygn** takes `example.png`, plots the image's pixel colors in the `RGB` color space and searches for `k=32` most dominant colors in it.
The image is then segmented and contour lines are produced, separating areas of different dominant colors. Since `n=0`, the matrix for removing noise in contour lines is not applied. Along the contour lines, each `d=10`th vertex is considered for triangulation. The resulting triangles are split into even smaller triangles when their area exceed `s=30`.

## Technical Explanation
coming soon

© llambdaa / Lukas Rapp 2022
