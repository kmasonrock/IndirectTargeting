# IndirectTargeting
These are the required scripts for the Indirect Targeting algorithm referenced in the paper "Indirect Targeting in Network Dynamcs".


If you want to make the gene regulatory networks, run "initialize_grn.jl".

Following this, use "grn_generation.jl" to make the networks, where the dynamical system can be found in "grn_generation.jl"

If you want to make food web networks, run "initialize_fw.jl".

Then, you can generate food webs using "fw_generation.jl", where the dynamical system can be found in "consumer_resource.jl".

Once networks have been generated, you can apply our perturbation algorithm using "scripts -> perturbations.jl". Please note, that we assign 5 workers to run in the current script for increased speed. Be aware that there may be a spike in memory usage when you run this! If you don't want to do that, there is a function that runs the indirect targeting algorithm on a single folder.


