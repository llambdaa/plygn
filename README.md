# plygn
<p align="center">
  Triangulation-based image abstraction for compression and artistic image generation written in Python.
<p align="center">
  This is a research project for my Bachelor's CS degree. Technically, I have to do it. But the topic was
  my choice and thus it was done with a lot of love! If you have any suggestions or ideas, feel free to
  tell me!

| :warning: | Contributions from others are not accepted for now. The project is still running formally and I later have to assure that I have done the work myself! After my studies, the project will be open for collaboration! |
|-----------|-----------|

## About
coming soon

## Installation
Before running the script, you must install multiple dependencies:
```
sudo pip3 install argparse opencv-python rawpy numpy numba faiss plotly
```

The first time you run the script on an example image, it will be slow.
That is, because it compiles some parts of it and caches them on your machine.
After that, it will run blazingly fast! At least... multiple times faster than before.

## Usage
The algorithm can be configured using program arguments. Some of them work in conjunction with each other
and amplify or damp their effects. You will understand them best if you have read about the inner workings.

 * `-i || --input`: Specifies image input path
 * `-o || --output`: Specifies image output path
 * `-c || --colorspace`: Specifies color space in which the image's pixel colors are plotted
 * `-k || --kmeans`: Specifies amount of dominant colors searched in color space
 * `-d || --distance`: Specifies the preferred distance between vertices along contour lines
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
