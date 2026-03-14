# Spectrophane

**Physics-based color lithophane generation**

Spectrophane explores a physics-based approach to generating **color lithophanes**.  
Instead of relying purely on empirical color mappings, the project models how light interacts with stacked translucent materials and uses this model to reproduce the colors of a target image.

The core idea is simple: if we understand the **optical behavior of the materials**, we should be able to **predict the resulting color** of a printed stack.

The project combines:

- optical simulation of layered materials  
- measured transmission and reflection spectra  
- machine-learning based parameter fitting 

The long-term goal is to make predictable and reproducible color lithophanes based on real material properties.

---

## Why a Physics-Based Approach?

Color mixing with **light-absorbing materials** is not directly predictable from their visible color.

A classic example illustrates the problem:

- In theory, mixing **blue and yellow absorbers** could produce **neutral grey**, depending on the exact spectral absorption curves.
- In practice, real paints usually produce **green**, because their spectra leave a window around intermediate (green) wavelengths where both materials absorb less light.

In other words:

> The visible color of a material does not uniquely determine how it mixes with other materials.

Generic subtractive color mixing in software therefore always relies on **empirical mappings or simplified subtractive models**.  
This can work reasonably well for fixed material sets, but they do not generalize reliably to new materials or arbitrary mixtures.

This limitation becomes relevant in **color lithophanes**, where **various real translucent materials** are stacked and interact with light in complex ways.

Spectrophane instead treats color formation as a **physical light transport problem**:

- materials are described by **optical properties**
- light transport through stacked layers is **simulated**
- the resulting spectrum is converted to color using **defined light sources and color matching functions**

The goal is to **predict perceived color from optical interactions**.

## Project Goals

Spectrophane currently focuses on two main goals.

### 1. Objectify Color Mixing

Replace heuristic or purely empirical color workflows with a **model grounded in the real optical behavior of materials**.

This enables:

- reproducible color results  
- more reliable simulation of stacked translucent materials  
- systematic exploration of new material combinations  

### 2. Lower the Barrier for Material Characterization

Accurate optical modeling normally requires specialized equipment such as:

- spectrometers  
- integrating spheres  
- calibrated optical setups  

Spectrophane explores workflows that make **useful material calibration possible without expensive laboratory equipment**.

Material parameters can be inferred from:

- spectral measurements **and/or**
- calibration images

This allows users to **extend the material palette themselves using only cameras**. 

---

## Material Calibration

Material parameters can be derived from:

- transmission spectra
- reflection spectra
- calibration images*

Spectrophane uses **gradient-based optimization** to infer optical parameters from these measurements.

The aim is not perfect physical characterization, but **parameters that are accurate enough to predict color behavior in layered prints**.

*As discussed above the user should be aware that there is ambiguity when calibrating from color values. To get the best results calibration images should include stacks with materials that are well characterized with spectral data.

---

## Who This Project Is For

Spectrophane may be interesting if you are:

### 3D Printing Enthusiasts

- generating more color accurate lithophanes with an open source tool

### Developers

- working with Python in scientific contexts
- interested in color science, resource optimization, or inverse problems

### Researchers / Engineers

- interested in simulating translucent materials

---

## Installation

Clone the repository and install the package in editable mode:

    pip install -e .

Then run the installation script:

    python3 src/spectrophane/scripts/install.py

This downloads the color science datasets required by the simulation.

### Optional Dependencies (for developers)

If you plan to contribute:

- install the optional Python dependencies defined in the project
- to remove personally identifying metadata from images before publishing, a git hook using **ExifTool** is included

Install ExifTool on Debian-based systems:

    sudo apt install libimage-exiftool-perl

---

## Example

A minimal example script for converting an image into a lithophane is included in:

    example_scripts/lithophane_creation.py

Run it after installing the package:

    python3 example_scripts/lithophane_creation.py

The script demonstrates the basic workflow:

- loading a target image
- loading material parameters from the included sample set  
  (Bambulab PLA CMYK Lithophane Bundle + black)
- computing a material stack representation
- exporting lithophane geometry

The script serves both as:

- a **starting point for experimentation**
- a **reference for interacting with the Spectrophane API**

Run

    spectrophane lithophane --help

for more details.

If you want to train your own parameter set, inspect:

    src/spectrophane/resources/training_data/default.json

and create a file in:

    [home directory]/Spectrophane/training_data/[your_file].json

Training can then be started with:

    spectrophane training [args]

Arguments are documented in:

    spectrophane training --help

---

## Conventions

### Material Stack Ordering

Material stacks follow a strict ordering convention:

**The last entry in a stack is on the observer side.**

Transmission:

    Light → first entry → ... → last entry → observer

Reflection:

    Light → last entry → ... → first entry → observer

In typical multi-material 3D printing workflows:

- the **first stack entry corresponds to the first printed layer**
- the **last entry corresponds to the front surface facing the viewer**

Keeping this convention consistent is important when defining materials and stacks.

---

## Project Status

Spectrophane currently exists as a **working prototype / early MVP**.

Core ideas and algorithms are implemented and usable, but several parts of the architecture are still evolving. Expect **refactoring and structural changes** as the project matures.

Current focus areas:

- validating the physics-based approach
- Reduce resource footprint of inversion and lithophane generation
- improving measurements for calibration of the sample set

---

## Contributing

Contributions are welcome, but the project is still in an exploratory phase.

Because parts of the codebase are evolving:

- APIs may change
- modules may be reorganized
- some approaches may be replaced

If you want to contribute:

1. Open an issue first to discuss your idea.
2. Smaller, focused contributions are easiest to integrate.
3. Calibration data, experiments, and testing with new materials are very welcome - especially if you have access to professional measurement equipment.

Feedback, bug reports, and results from real prints are highly appreciated.

---

## License

The Spectrophane codebase is distributed under the **MIT License**.

The project downloads and uses **illuminant and observer/color matching datasets** provided by the CIE.

These datasets are:

© CIE, licensed under  
**Creative Commons Attribution-ShareAlike 4.0 International**

Details:  
https://creativecommons.org/licenses/by-sa/4.0/

Files downloaded by the installation script are listed in:

    src/spectrophane/scripts/install.py

---

## Inspiration and Honorable Mentions

Many ideas in this project build on previous work:

- PIXEstL: Open-source color lithophane generator. Investigating color mixing limitations there led to the idea for this project.
  https://github.com/gaugo87/PIXEstL

- Lars Wander: Demonstrated machine-learning based fitting of Kubelka–Munk spectra for virtual pigment mixing.
  https://larswander.com/writing/spectral-paint-curves/

- Mixbox: Realistic pigment mixing for digital art using a fixed set of base pigment data. Also inspired the LUT inversion approach used here.
  https://github.com/scrtwpns/mixbox