# Fracture segmentation assisted with superpixels


**Paper: Interactive fracture segmentation based on optimum connectivity between superpixels**

*Authors: Ademir Marques Junior, Graciela Racolte, Eniuce Menezes de Souza, Leonardo Bacchi, Alexandre Falcao, Caroline Lessio Cazarin, Luiz Gonzaga Jr, Mauricio Roberto Veronez

Abstract: Oil and gas reservoirs are well studied in petroleum engineering using seismic data to estimate fluid flow and well placement. However, seismic data cannot capture fractures due to their scale, and fractures may affect rock porosity and permeability. Consequently, rock fracture segmentation and quantification from aerial images of analogous outcrops can input essential information into those studies. This paper addresses the fracture segmentation problem through the Image Foresting Transform (IFT) -- a tool for the design of image operators based on optimum connectivity. First, the image is segmented into superpixels. By defining a superpixel graph and user-selected seed superpixels, fractures can be delineated by one of two proposed IFT algorithms. This strategy considerably increases computational efficiency while reducing human effort in fracture segmentation compared to manual annotation in a pixel-by-pixel fashion. We evaluate our algorithms with multiple specialists using the same set of 20 images to account for bias among human interpretations and to generate a dataset with consolidated annotation for future work. The interpretation (segmentation) time among users fluctuated between 80 and 200 minutes for the entire set, and the average F1 score ranged between 0.86 and 0.93, showing significant variation among users while demonstrating a valid showcase of the proposed algorithms.

Work sent to the SIBIGRAPI 2022 - 35th Conference on Graphics, Patterns and Images


# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)
- [How to cite](#how-to-cite)


## Installation

This Python script requires a Python 3 environment and the following installed libraries as seen in the file requirements.txt.

- [PyQt5](https://keras.io/)
- [Numpy](https://numpy.org/)
- [Pillow]()
- [Numba](https://pypi.org/project/sewar/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [Pillow]()


For installation download and unpack the content of this repository in a folder acessible by your working Python 3 environment.


## Usage

Run the script "main.py" in a Python 3 environments with the required libraries installed to open the main window.

<img src="https://github.com/ademirmarquesjunior/superpixel_segmentation/blob/main/docs/screen.png" width="500" alt="Neural network learning process">

The menu item "Files" will open option to open the image to be segmented, to open the superpixel image, and to save the segmented mask.

<img src="https://github.com/ademirmarquesjunior/superpixel_segmentation/blob/main/docs/open_files.png" width="150" alt="Neural network learning process">

The file examples below are in the "/Dataset/Images + Superpixel/" folder use examples, where the tiff images have to be openned first followed to the "superpixel" file.

<img src="https://github.com/ademirmarquesjunior/superpixel_segmentation/blob/main/docs/files.png" width="150" alt="Neural network learning process">

Conducted experiments with three geologists and the consolidated dataset generated in the paper above are in the "/Dataset/" folder.



## Credits	
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://vizlab.unisinos.br/) and the following developers:	[Ademir Marques Junior](https://www.researchgate.net/profile/Ademir_Junior) and Alexandre Falcão (Unicamp).

## License

    MIT Licence (https://mit-license.org/)

## How to cite

If you find our work useful in your research please consider citing one of our papers:

```bash
To be published
```

