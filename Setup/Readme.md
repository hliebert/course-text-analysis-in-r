# Setup instructions

1. **Install R** 

   You can install R by downloading  [the installer from the
   website](https://cran.r-project.org/) (on Windows), or via your system's
   package manager (e.g. `homebrew` on MacOS or `apt` on Ubuntu/Debian Linux). 
   
   Alternatively, you can install R through `conda` after installing the
   Anaconda distribution (or its smaller Miniconda version). Anaconda provides
   Python, R, and a repository hosting most of the libraries for both languages.
   Download links for different operating systems are found
   [here](https://docs.anaconda.com/anaconda/install/), documentation and
   instructions  [here](https://conda.io/projects/conda/en/latest/).

2. **Install an R GUI or text editor with R support** 

   Plugins are available for most editors (VS Code, Emacs, Vim, Atom, ...).
   I recommend using one of the following if you are starting out. 
      - [Rstudio](https://www.rstudio.com/products/rstudio/)
      - [VS Code with R plugin](https://code.visualstudio.com/)

   All code is provided as simple text files (suffix `.r`) and as Jupyter
   notebooks using the R kernel (suffix `.ipynb`). I am using the notebooks for
   didactic purposes only. You do not need to use them to follow the course. 
   
   However, if you want access to Jupyter notebooks, you need to install
   Anaconda (see above), or install Python and then the `jupyter` package using
   the `pip` package manager. If you are using Windows and are unsure what a
   package manager is, I recommend installing Anaconda. 
 
3. **Install package dependencies** 

   If you use a native R installation (e.g., from the R project website), just run
   the contents of the `Setup/install.r` file provided. 
   
   If you use `conda`, import the file `Setup/environment.yml` using the GUI.
   Alternatively, run the following commands in a terminal or the Anaconda
   console (on Windows)  to create the environment and  to activate it.

   ```
   conda env create --file Setup/environment.yml 
   conda activate course-text-analysis
   ```

4. **Troubleshooting** 
   
   If you run into trouble during installation, please [contact
   me](mailto:helge.liebert@econ.uzh.ch). Supporting all possible edge cases on
   different operating systems is difficult. If everything fails, all lab
   material  can be also be run in your browser using the links below.

   [![Binder](https://mybinder.org/badge_logo.svg) Jupyter Notebook](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=tree)  
   [![Binder](https://mybinder.org/badge_logo.svg) Jupyter Lab](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=lab)  
   [![Binder](https://mybinder.org/badge_logo.svg) Rstudio](https://mybinder.org/v2/gh/hliebert/course-text-analysis-in-r/HEAD?urlpath=rstudio)  
