# Protein Superposition by Gesamnt 
Given an UniProt ID, GesamtImplementation.py downloads all the pdbs for that UniProt ID and superposes them in a single pdb file. 

It reproduces both the "3D view of superposed structures" with and without ligands for a given uniprot. Example: [O94788](https://www.ebi.ac.uk/pdbe/pdbe-kb/proteins/O94788).

## Example of usage
```bash
python3 path/to/GesmantProteinSuperposition.py --path2analysis_folder /path/where/results/stored --path2gesamt /path/to/gesamt/executable --uniprot_input O14746,O14842,O14965
```

## Disclaimer
The order of the cluster does not necesarily match the one given by pdbe-kb webpage, that is, local cluster 0 could correspond with cluster 3 from the pdbe-kb view. 


