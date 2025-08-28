# Protein Superposition by Gesamnt 
Given an UniProt ID, GesamtImplementation.py downloads all the pdbs for that UniProt ID and superposes them in a single pdb file. 

It reproduces both the "3D view of superposed structures" with and without ligands for a given uniprot. Example: [O94788](https://www.ebi.ac.uk/pdbe/pdbe-kb/proteins/O94788).

## Example of usage
```bash
python3 path/to/GesmantImplementation.py --path2analysis_folder /path/where/results/stored --uniprot_input O94788,034926
```

## Requirements 
- numpy
- pandas
- Biopython
- [protein-cluster-conformation](https://github.com/PDBeurope/protein-cluster-conformers)

## Disclaimer
- The order of the cluster does not necesarily match the one given by pdbe-kb webpage, that is, local cluster 0 could correspond with cluster number 3 from the pdbe-kb view.
- In the clustering process sometimes errors arise due to chain missmatches. This is most likely due to gesamt issues.
- There might be some issue do to numpy version nan import. Just go to the protein-cluster-conformation file that throws the error and change the import to -> from numpy import nan as NaN



