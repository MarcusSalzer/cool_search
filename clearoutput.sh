#!/bin/bash

source ~/_my_python/bin/activate

# Find all Jupyter notebooks in the repository
notebooks=$(ls scripts/ | grep '\.ipynb$')

# Loop over each notebook and clear its output
for notebook in $notebooks; do
    echo "Clearing output from scripts/$notebook"
    jupyter nbconvert --clear-output --inplace "scripts/$notebook"
done
