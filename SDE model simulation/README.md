# This is the SDE model used in the paper.

The SDE model use in this study is the system describing the MYC/E2F pathway in cell-cycle progression. Upon growth factor stimulation, increases in MYC lead to activation of E2F-regulated genes through two routes. First, MYC regulates expression of Cyclin D, which serves as the regulatory components of kinases that phosphorylate pocket proteins and disrupt their inhibitory activity. Second, MYC facilitates transcriptional induction of activator E2Fs, which activate the transcription of genes required for S phase. Expression of activator E2Fs is reinforced by two positive feedback loops. First, activator E2Fs can directly binds to their own regulatory sequences to help maintain an active transcription state, second, activator E2Fs transcriptionally upregulate CYCE, which stimulates additional phosphorylation of pocket proteins and prevents them from sequencing activator E2Fs. 
To capture stochastic aspects of the Rb-E2F signaling pathway, we adopted the Chemical Langevin Formulation (CLF). We adjusted the units of the molecule concentrations and the parameters so that the molecules are expressed in molecular numbers. 