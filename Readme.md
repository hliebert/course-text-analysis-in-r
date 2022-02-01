# Text Analysis in R

This repository contains material for the course Text Analysis in R during the
2022 Winter School in Data Analytics and Machine Learning at the Université de
Fribourg. The lab material is set up to run on Binder.

[![Binder](https://mybinder.org/badge_logo.svg) Jupyter Notebook](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=tree)  
[![Binder](https://mybinder.org/badge_logo.svg) Jupyter Lab](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=lab)  
[![Binder](https://mybinder.org/badge_logo.svg) Rstudio](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=rstudio)  


# Setup instructions

1. **Install R** 

   You can install R by downloading [the installer from the
   website](https://cran.r-project.org/) (on Windows), or via your system's
   package manager (e.g. `homebrew` on MacOS or `apt` on Ubuntu/Debian Linux).
   On Windows, you will also need to install the [Rtools
   toolchain](https://cran.r-project.org/bin/windows/Rtools/rtools40.html).
   
   Alternatively, you can install R through `conda` after installing the
   Anaconda distribution (or its smaller Miniconda version). Anaconda provides
   Python, R, and a repository hosting most of the libraries for both languages.
   Download links for different operating systems are found
   [here](https://www.anaconda.com/products/individual#Downloads), documentation
   and instructions [here](https://conda.io/projects/conda/en/latest/).

2. **Install an R GUI or text editor with R support** 

   Plugins are available for most editors (VS Code, Emacs, Vim, Atom, ...).
   I recommend using one of the following if you are starting out. 
      - [RStudio](https://www.rstudio.com/products/rstudio/)
      - [VS Code with R plugin](https://code.visualstudio.com/)

   All code is provided as simple text files (suffix `.r`) and as Jupyter
   notebooks using the R kernel (suffix `.ipynb`). I am using Jupyter for
   didactic purposes only. You do not need to use the notebooks to follow the
   course. 
   
   However, if you want access to Jupyter notebooks, you need to install
   Anaconda (see above), or install Python and then the `jupyter` package using
   the `pip` package manager [(instructions here)](https://jupyter.org/install).
   If you are using Windows and are unsure what a package manager is, I
   recommend installing Anaconda. 
 
3. **Install required R libraries** 

   Installation files are provided in the folder `Setup`. If you use a native R
   installation (e.g., from the R project website), just run the contents of the
   `install.r` file provided. On MacOS and Linux, you may need to install
   additional dependencies on your system (I recommend using
   [Homebrew](https://brew.sh/) for this on MacOS). The error messages during
   the installation will typically point you towards the solution.
   
   If you use Anaconda, import the file `environment.yml` using the GUI.
   Alternatively, run the following commands in a terminal or the Anaconda
   console (on Windows) to create the environment and to activate it.

   ```
   conda env create --file Setup/environment.yml 
   conda activate course-text-analysis
   ```

4. **Troubleshooting** 
   
   If you run into trouble during installation, please [contact
   me](mailto:helge.liebert@econ.uzh.ch). Supporting all possible edge cases on
   different operating systems is difficult. If all else fails, simply run the
   lab material in your browser using the links below.

   [![Binder](https://mybinder.org/badge_logo.svg) Jupyter Notebook](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=tree)  
   [![Binder](https://mybinder.org/badge_logo.svg) Jupyter Lab](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=lab)  
   [![Binder](https://mybinder.org/badge_logo.svg) Rstudio](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=rstudio)  
