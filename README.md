# Khata

Khata is a pre-commit Git hook that runs on the change that is upon submission to evaluate how risky the change is.

## Setup

To setup Khata, developers need to place the files in this repository under directory `.git/hooks/` in their projects. As long as `pre-commit` file is under the mentioned directory, Khata runs every time `git commit` command is run and displays the probabilty of the change being defective.

## Extending Khata

Currently, the model Khata is using to predict defectiveness of commits is JITGNN, a graph neural network model trained on thousands of commits to predict the probability of a given commit being defective. To extend Khata with other models, one need to build a subclass from `HookInterface` in file `hook_interface.py`. `HookInterface` contains the files that are modified in attribute `modifieds` and any subclass from this class has access to this attribute. 

This class has an abstract method called `run_model` that should be implemented based on the new models that are extending Khata. This method essentially displays the probability of the commit being defective and **exits with value 0** (`sys.exit(0)`). Exiting with any value other than zero aborts the commit which is not the desired behavior.