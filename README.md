## Cite Paper
Kapusuzoglu, B., Nath, P., Sato, M., Mahadevan, S., & Witherell, P. (2021). Multi-Objective Optimization Under Uncertainty of Part Quality in Fused Filament Fabrication. ASCE-ASME J Risk and Uncert in Engrg Sys Part B Mech Engrg.

Please, cite this repository using: 
    
    @article{kapusuzoglu2021multi,
    author = {Kapusuzoglu, Berkcan and Nath, Paromita and Sato, Matthew and Mahadevan, Sankaran and Witherell, Paul},
    title = "{Multi-Objective Optimization Under Uncertainty of Part Quality in Fused Filament Fabrication}",
    journal = {ASCE-ASME J Risk and Uncert in Engrg Sys Part B Mech Engrg},
    year = {2021},
    month = {12},
    issn = {2332-9017},
    doi = {10.1115/1.4053181},
    url = {https://doi.org/10.1115/1.4053181},
    eprint = {https://asmedigitalcollection.asme.org/risk/article-pdf/doi/10.1115/1.4053181/6806801/risk-21-1006.pdf},
}

# Multi-Objective Optimization Under Uncertainty of Part Quality in Fused Filament Fabrication

# Abstract
This work presents a data-driven methodology for multi-objective optimization under uncertainty of process parameters in the fused filament fabrication (FFF) process. The proposed approach optimizes the process parameters with the objectives of minimizing the geometric inaccuracy and maximizing the filament bond quality of the manufactured part. First, experiments are conducted to collect data pertaining to the part quality. Then, Bayesian neural network (BNN) models are constructed to predict the geometric inaccuracy and bond quality as functions of the process parameters. The BNN model captures the model uncertainty caused by the lack of knowledge about model parameters (neuron weights) and the input variability due to the intrinsic randomness in the input parameters. Using the stochastic predictions from these models, different robustness-based design optimization formulations are investigated, wherein process parameters such as nozzle temperature, nozzle speed, and layer thickness are optimized under uncertainty for different multi-objective scenarios. Epistemic uncertainty in the prediction model and the aleatory uncertainty in the input are considered in the optimization. Finally, Pareto surfaces are constructed to estimate the trade-offs between the objectives. Both the BNN models and the effectiveness of the proposed optimization methodology are validated using actual manufacturing of the parts.

DEAP and platypus are used to do the multi-objective optimizations.

# Notebooks
## ParetoOptim_AM_DEAP_3var
Performs a pareto front optimization for 3 integer variables (printer temperature, speed and layer height).

## ParetoOptim_AM_DEAP_3var_Exp_Design
Since evolutionary algorithms are stochastic algorithms, therefore results of *ParetoOptim_AM_DEAP_3var* must be assessed by repeating experiments until a statistically valid conclusion is reached by using *hypervolume* computation.

## ParetoOptim_AM_DEAP_3var_Exp_Animation
Save the animation of the Pareto optimal fronts

## ParetoOptim_AM_DEAP_3var_Exp_Hypervolume
Calculate the hypervolume indicator wrt. number of generations


## ParetoOptim_AM_DEAP_3var
Performs a pareto front optimization for 3 integer variables (printer temperature, speed and layer height).
Objective functions are BL (overall mean and standard deviation of the bond length, and thickness)

## MOO_Case1-Copy1
Multi-objective-optimization of case 1. Pareto front plotted.

## MOO_Case1-Copy1_MCS
Multi-objective-optimization of case 1. Pareto front plotted with MCS results.
