Spectrophane aims to bring physics based color mixing to color Lithophane generation. This is done by deriving material properties from transmission/reflection spectra and photographic data by machine learning, creating color palettes for material combinations and finally translating image data to best fitting material stacks forming the lithophane. This project is still under construction and not functional.

Installation:
Open a terminal in the root directory of the project (where this file lives). Run 
pip install . -e 
. then run 
python3 /src/Spectrophane/scripts/install.py
If you want to contribute you will need to install optional packages (see pyproject.toml). If you want to contribute images in the raw format you will need to install exiftools (e.g. apt install libimage-exiftool-perl)


Conventions:
Material stack order: The last entry in a stack is on the side of the observer. In Transmission this means light hits the first entry and exits the last, in reflection light hits the last and exits the last entry. Typically the first entry corresponds to the first layer in a 3D print.

LICENCE:
This code is distributed under the MIT licence.
The project downloads and utilizes illuminator and observer/color matching functions from the CIEs website. The data are © CIE and licenced under Creative Commons Attribution-ShareAlike 4.0 International License. See https://creativecommons.org/licenses/by-sa/4.0/ for details. Downloaded files are listed in src/spectrophane/scripts/install.py
