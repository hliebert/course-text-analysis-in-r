# Text Analysis in R

This repository contains material for the course Text Analysis in R during the
2022 Winter School in Data Analytics and Machine Learning at the Université de
Fribourg.

The lab material is set up to run on Binder.

Jupyter Notebook [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=tree)  
Jupyter Lab [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=lab)  
Rstudio [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=rstudio)  


# Setup instructions

1. **Install R.** 

   Either from the website ([https://cran.r-project.org/]), or via your system's
   package manager (e.g. `homebrew` on MacOS or `apt` on Ubuntu/Debian Linux).
   Alternatively you can install R through Anaconda/Miniconda
   ([https://conda.io/projects/conda/en/latest/]). Download links for your
   operating system are found here:
   ([https://docs.anaconda.com/anaconda/install/]).

2. **Install an R GUI or text editor with R support** 

   I recommend using either of the following. Plugins are available for most
   editors (VS Code, Sublime, Atom, Emacs, Vim, ...).
  - [Rstudio](https://www.rstudio.com/products/rstudio/)
  - [VS Code with R plugin](https://code.visualstudio.com/)

  All code is provided as simple text files (`.r`) and as Jupyter notebooks
  using the R kernel (`.ipynb`). I am using the notebooks for didactic purposes
  only. You do not need them to follow the course. 
  
  However, if you want access to Jupyter notebooks, you need to install Anaconda
  (see above), or install Python and then the `jupyter` package using the `pip`
  package manager. If you are using Windows and are unsure what a package
  manager is, I recommend installing Anaconda. 
 
3. **Install package dependencies*** 

   If you use a native R installation (e.g., from the R project website), just run
   the contents of the `Setup/install.r` file provided. 
   
   If you use `conda`, import the file `environment.yml` using the GUI.
   Alternatively, run the following commands in a terminal or the Anaconda
   console (on Windows)  to create the environment and  to activate it.

   ```
   conda env create --file environment.yml 
   conda activate course-text-analysis
   ```

4. **Troubleshooting*** 
   
   If you run into trouble during installation, please (contact
   me)[mailto:helge.liebert@econ.uzh.ch]. Supporting all possible edge cases on
   different operating systems is difficult. If everything fails, all lab
   material  can be also be run in your browser using the links below.

   Jupyter Notebook [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=tree)  
   Jupyter Lab [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=lab)  
   Rstudio [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=rstudio)  
