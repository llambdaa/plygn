# StudienarbeitImpl
Triangulation-based image abstraction for compression and artistic image generation written in Python.

## Installation
Before running the script, you must install multiple dependencies:
```
sudo pip3 install argparse opencv-python rawpy numpy numba faiss plotly
```

The first time you run the script on an example image, it will be slow.
That is, because it compiles some parts of it and caches them on your machine.
After that, it will blazing fast! At least... multiple times faster than before.
