# PDE_model
 This is the matlab code for generating the PDE model for synthetic cell pattern formation


The main matlab file is "wrapper_parameter_scan_SW.m"

After the data generated, clean and sort the data into a csv file. We include a sample csv file called "all_data.csv" with only 1,000 simulation results

If want to run the code on SLURM clusters, please use: 
sbatch wrapper_parameter_scan_SW.q


The circuit consists of a mutant T7 RNA polymerase (T7RNAP) that activates its own gene expression and the expression of LuxR and LuxI. LuxI synthesizes an acyl-homoserine lactone (AHL) which can induce expression of T7 lysozyme upon binding and activating LuxR. Lysozyme inhibits T7RNAP and its transcription by forming a stable complex with it. CFP and mCherry fluorescent proteins are used to report the circuit dynamics since they are co-expressed with T7RNAP and lysozyme respectively. 
The PDE model used in the current study corresponds to the hydrodynamic limit of the stochastic agent-based model from Payne et al.. Because the air pocket between glass plate and dense agar is only 20 um high, the system was modeled in two spatial dimensions and neglect vertical variations in gene expression profiles42. Although the PDE formulation is computationally less expensive to solve numerically than the stochastic agent-based model and better facilitates development of mechanistic insights into the patterning dynamics, it still needs a lot of computational power when extensive parameter search is needed.


