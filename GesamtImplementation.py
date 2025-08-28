#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025

@author: andres
"""

import os 
import pandas as pd
import requests
import subprocess
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
import Bio.PDB as bpdb
from Bio import PDB
from Bio.PDB.NeighborSearch import NeighborSearch
import numpy  as np
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from cluster_conformers.utils.download_utils import fetch_updated_mmcif
from cluster_conformers.cluster_monomers import ClusterConformations
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from itertools import combinations
import argparse
import datetime


substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', '5OW':'LYS', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MK8':'LEU', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

solvents = 'EOH, MOH, GOL, 1GP, 2DP, 3PH, 6A8, DGA, DGD, DGG, DR9, DVG, G3P, HGP, \
            HGX, IGP, INB, L1P, L2P, L3P, L4P, LHG, LI1, LIO, LPC, PGM, SGL, SGM, SQD, TGL, \
            12P, 15P, 1PE, 2PE, CE9, CP4, DIO, P4C, P6G, PG4, PGE, VNY, DMS, EDO, PEG, TRS, \
            IPA, TBU, ACT, EEE, ACY, BME, MBN, NAG, SIA, FUK, IVA, STA, BMA, SO4, MAN, GAL, \
            DOD, SO3, IOD, PO4, PO3, TLA, PSA, MES, PG4, FUC, SOG, GLC, SF4'
solvents = [i.strip() for i in solvents.split(',')]

class ChainSelect(bpdb.Select):
    def __init__(self, chains2keep, struct):
        self.chains2keep = chains2keep 
        atoms_prot = [atom for residue in struct.get_residues() if residue.id[0] == ' ' and residue.get_full_id()[2] in self.chains2keep for atom in residue]
        self.ns = NeighborSearch(atoms_prot)
        
    def accept_residue(self, res):
        full_id = res.get_full_id()
        
        if full_id[3][0].startswith('H_'):
            close_res_prot= []
            for atom in res:
                close_res_prot.extend(self.ns.search(atom.coord, 5, level="A"))
            
            if len(close_res_prot) > 0:
                return True
            else:
                return False
        
        if full_id[3][0] == ' ' and full_id[2] in self.chains2keep:
            return True
        
        else:
            return False

def build_updated_uniprotpdb_dataset(path2pdbchainuniprot_tsv,
                                     pdb_column_name = 'PDB', 
                                     uniprot_column_name='SP_PRIMARY'):
    
    """  Tsv downloaded from https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html
    Table name file: pdb_chain_uniprot.tsv.gz
    # https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html
    The version and date can be checked within the file"""
    
    url = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(path2pdbchainuniprot_tsv, 'wb') as file:
            file.write(response.content)
        print(f"File successfully downloaded as {path2pdbchainuniprot_tsv}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        return None
    
    df_org = pd.read_csv(path2pdbchainuniprot_tsv, sep = '\t', low_memory=False, header=1)
    
    df = df_org[[pdb_column_name, uniprot_column_name]]
    df = df.drop_duplicates()

    df_exploded = df.assign(PDB=df[pdb_column_name].str.upper().str.split()).explode(pdb_column_name)
    df_grouped = df_exploded.groupby(pdb_column_name)[uniprot_column_name].agg(lambda x: '-'.join(sorted(set(x)))).reset_index()
    df_final = df_grouped.groupby(uniprot_column_name)[pdb_column_name].agg(lambda x: ' '.join(sorted(x))).reset_index()
    
    path2folder = "/".join([i for i in path2pdbchainuniprot_tsv.split('/')[:-1]])
    df_org[['PDB', 'CHAIN', 'SP_PRIMARY']].to_csv(path2folder+'/uniprot2pdb_raw.csv', index = None)
    df_final.to_csv(path2folder+'/uniprot2pdb.csv', index = None)

    return path2folder+'/uniprot2pdb_raw.csv', path2folder+'/uniprot2pdb.csv'

def reorder_pdbs_by_residue_count(pdb_list, path2save):

    parser = PDBParser(QUIET=True)
    
    pdb_counts = []
    for pdb_file in pdb_list:
        structure = parser.get_structure(pdb_file, path2save + pdb_file)
        residue_count = sum(1 for model in structure for chain in model for res in chain if res.id[0] == ' ')
        pdb_counts.append((pdb_file, residue_count))

    sorted_pdbs = sorted(pdb_counts, key=lambda x: x[1], reverse=True)
    
    return [pdb[0] for pdb in sorted_pdbs]

def uniprot_superimposer_clustering(uniprot_id, uniprot_exact_match=True, prot_format = '.pdb'):
    
    path2save = f"{analysis_folder}{uniprot_id}"
    if not os.path.exists(path2save):
        os.mkdir(path2save)
    
    uniprot_ids = uniprot_id.split('-')
    
    all_pdbs = []
    for u_i in set(uniprot_ids):
        all_pdbs.append(uniprot2pdb[uniprot2pdb.SP_PRIMARY.apply(lambda x: u_i in x.split('-'))])
    
    all_pdbs_df = pd.concat(all_pdbs, ignore_index=True)
    all_pdbs = all_pdbs_df.PDB.str.upper().str.split().values.flatten().tolist()
    all_pdbs = [j for i in all_pdbs for j in i]
    
    path_save_ummcifs = Path(
        "updated_mmcifs",
        path2save
    )
        
    prev_input_dictionary = dict()
    check_dict = []
    for nn, p2d in enumerate(all_pdbs):

        fetch_updated_mmcif(
            pdb_code=p2d.lower(),
            path_save=path_save_ummcifs
        )
        
        df_chain = uniprot2pdb_raw[(uniprot2pdb_raw.PDB == p2d.lower()) & (uniprot2pdb_raw.SP_PRIMARY == uniprot_id)]
        uniprot_chains = df_chain.CHAIN.values.flatten().tolist()
        struct = MMCIFParser(QUIET=True, auth_chains=True).get_structure('balbla', path2save+f'/{p2d.lower()}_updated.cif')

        present_chains = [i.id for i in struct.get_chains() if any(res.get_resname() in substitutions.values() for res in i) and i.id in uniprot_chains]
        prev_input_dictionary[path2save+'/'+f'{p2d.lower()}_updated.cif'] = present_chains
        check_dict.extend([f'{p2d.lower()}_{i}' for i in present_chains])
        print(f'{nn+1}/{len(all_pdbs)} _updated.cifs Downloaded') 
    
    print("Download completed.")
    
    if len(all_pdbs) < 2:
        return [f'{uniprot_id} No or 1 pdbs']
              
    input_dictionary = prev_input_dictionary
    intercalated_cif = [f'-m {k} {" ".join(v)}'.split(' ') for k, v in input_dictionary.items() if all([len(w) == 1 for w in v])]
    intercalated_cif = [j for i in intercalated_cif for j in i]
    paths2command = ['-c', f'{path2save}/{uniprot_id}_ca_distances', 
                     '-d', f'{path2save}/{uniprot_id}_distance_differences/',
                     '-s', f'{path2save}/{uniprot_id}_cluster_results/']
    findconformers_comand = ['python3', '/home/andres/Desktop/protein-cluster-conformers/find_conformers.py', '-u', uniprot_id] + intercalated_cif + paths2command
    subprocess.run(findconformers_comand, check = True)

    for prot in [i for i in os.listdir(path2save) if i.endswith('.cif')]:
        struct = MMCIFParser(QUIET=True, auth_chains=False).get_structure('balbla', path2save+'/'+prot)
        io=MMCIFIO()
        io.set_structure(struct)
        io.save(path2save+'/'+prot, preserve_atom_numbering=True)

    df_cluster = pd.read_csv(f'{path2save}/{uniprot_id}_cluster_results/{uniprot_id}_sum_based_clustering_results.csv') #.drop_duplicates().groupby(['CONFORMER_ID'])
    
    res_len = df_cluster.drop_duplicates().apply(lambda x: len([i for i in MMCIFParser(QUIET=True, auth_chains=False).get_structure(x.loc['PDBe_ID'], path2save+'/'+ f'{x.loc["PDBe_ID"].lower()}_updated.cif').get_residues() if i.get_parent().id == x.loc['CHAIN_ID']]), axis = 1)
    df_lengths = pd.DataFrame({'PDBe_ID': df_cluster['PDBe_ID'].drop_duplicates(), 'res_len': res_len})
    df_cluster = df_cluster.merge(df_lengths, on = 'PDBe_ID', how = 'inner')
    df_cluster = df_cluster.drop_duplicates().sort_values(by = 'res_len', ascending = False).groupby(['CONFORMER_ID'])

    for index, group in df_cluster:
        index = np.unique(group['CONFORMER_ID'])[0]
        print(group, index)
        if len(group) == 1:
            superimposed_st = MMCIFParser(QUIET=True).get_structure('blabla', path2save+'/'+group['PDBe_ID'].values.tolist()[0]+'_updated.cif')
            io=PDBIO()
            io.set_structure(superimposed_st)
            io.save(path2save+f'/superimposed_all_cluster_{index}.pdb', ChainSelect(group['CHAIN_ID'].tolist(), superimposed_st),
            preserve_atom_numbering=True)
            continue 
        
        addcomands = [path2save + f'/{p}_updated.cif -s {c}' for p, c in zip(group.PDBe_ID, group.CHAIN_ID)]
        addcomands = [j for i in addcomands for j in i.split()]
        gesamt_command = [path2gesamt] + addcomands + ['-o', path2save + f'/superimposed_{uniprot_id}_cluster_{index}.pdb', '-csv', path2save + f'/transmatrix_{uniprot_id}_cluster_{index}.csv', '-nthreads=auto']
        gesamt_message = subprocess.run(gesamt_command, check = True, capture_output=True, text =True) # 
        # ver = gesamt_message.stdout
        while '0 atoms selected' in gesamt_message.stdout:
            gesamt_split = gesamt_message.stdout.split('\n')
            todelete = []
            for n_l, line in enumerate(gesamt_split):
                if '0 atoms selected' in line:
                    todelete.append(gesamt_split[n_l-1].split("'")[1])
                    index2delete = gesamt_command.index(gesamt_split[n_l-1].split("'")[1])
                    [gesamt_command.pop(index2delete) for n in range(3)]
                    gesamt_message = subprocess.run(gesamt_command, check = True, capture_output=True, text =True)
                    print('deleted', index)
                    
        gesamt_command_pdb = [i.split('/')[-1] for i in gesamt_command if i.endswith('.cif') or len(i) == 1]
        with open(path2save + f'/superimpose_pdb_mapping_cluster_{index}.txt', mode = 'w') as file_map:
            index1 = 1
            for w, pp in enumerate(gesamt_command_pdb):
                if pp.endswith('.cif'):
                    file_map.write(f'Model_{index}_{pp[:4]}_{gesamt_command_pdb[w+1]}\n')
                    index1 += 1
         
        if len(group['PDBe_ID'].tolist()) == 1:
            structure_prot =  MMCIFParser(QUIET=True).get_structure(p2d,path2save+'/'+ f'{group["PDBe_ID"][0]}_updated.cif')

            io = PDBIO()
            io.set_structure(structure_prot)
            io.save(path2save+f'/superimposed_all_cluster_{index}.pdb', ChainSelect(group['CHAIN_ID'].tolist(), structure_prot),
            preserve_atom_numbering=True)
            print(f'Saved Cluster {index} to {path2save}/superimposed_all_cluster_{index}.pdb')
            # return [uniprot_id, 'only 1 cluster out of', len(df_cluster)]
            
            
        if len(group['PDBe_ID'].tolist()) == 2:
            df_trans = pd.DataFrame(columns = ['pdb_id', 'trans_mat'])
            for ppu in group['PDBe_ID'].tolist():
                with open(path2save + f'/transmatrix_{uniprot_id}_cluster_{index}.csv') as file_mat:
                    for line in file_mat:
                        if 'Rx' in line:
    
                            row1 = next(file_mat)
                            row2 = next(file_mat)
                            row3 = next(file_mat)
                            
                            row1 = list(map(float, row1.strip().split(',')))
                            row2 = list(map(float, row2.strip().split(',')))
                            row3 = list(map(float, row3.strip().split(',')))
                            break
                    df_trans.loc[len(df_trans)] = ppu, np.asarray([row1,row2,row3])
        
        if len(group['PDBe_ID'].tolist()) > 2:    
            df_trans = pd.DataFrame(columns = ['pdb_id', 'trans_mat'])
            for ppu in group['PDBe_ID'].tolist():
                with open(path2save + f'/transmatrix_{uniprot_id}_cluster_{index}.csv') as file_mat:
                    for line in file_mat:
                        if ppu in line and 'STRUCTURE' in line:
                            _ = next(file_mat)
                            _ = next(file_mat)
                            
                            row1 = next(file_mat)
                            row2 = next(file_mat)
                            row3 = next(file_mat)
                            
                            row1 = list(map(float, row1.strip().split(',')))
                            row2 = list(map(float, row2.strip().split(',')))
                            row3 = list(map(float, row3.strip().split(',')))
                            break
                    df_trans.loc[len(df_trans)] = ppu, np.asarray([row1,row2,row3])
        
        superimposed_st = PDBParser(QUIET=True).get_structure('superimposed', path2save + f'/superimposed_{uniprot_id}_cluster_{index}.pdb')
        for ppu, model_su in zip(group['PDBe_ID'].tolist(), superimposed_st):
            if ppu == group['PDBe_ID'].tolist()[model_su.id]:
                structure_prot =  MMCIFParser(QUIET=True).get_structure(p2d, path2save+'/'+ f'{ppu}_updated.cif')
                for model in structure_prot:
                    superimposed_st_residues = [j.get_id() for i in superimposed_st for j in i.get_residues() if j.get_id()[0].startswith('H_')]
                    for chain in model:
                        for residue in chain:
                            if residue.get_id()[0].startswith("H_") and residue.get_id() not in superimposed_st_residues and residue.get_id()[0][2:] not in solvents and residue.get_id()[0][2:] not in substitutions:
                                print(residue.get_id())
                                if chain.id[0] not in model_su:
                                    model_su.add(Chain.Chain(chain.id[0]))
                                superimposed_chain = model_su[chain.id[0]]
                                new_residue = Residue.Residue(residue.id, residue.resname, residue.segid)
                                for atom in residue:
                                    pos_org = np.append(atom.coord, 1)  
                                    pos_trs = np.dot(df_trans[df_trans.pdb_id == ppu]['trans_mat'].values[0], pos_org)
                                    atom.coord = pos_trs[:3]
                                    new_residue.add(atom.copy())
                                superimposed_chain.add(new_residue)
            else:
                print('OJO CUIDAO')
        
        io = PDBIO()
        io.set_structure(superimposed_st)
        io.save(path2save+f'/superimposed_all_cluster_{index}.pdb', ChainSelect(group['CHAIN_ID'].tolist(), superimposed_st),
        preserve_atom_numbering=True)
        print(f'Saved Cluster {index} to {path2save}/superimposed_all_cluster_{index}.pdb')
      


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Protein Superposition for all the pdbs of a UniProt ID by Gesamt')
    
    parser.add_argument('--path2analysis_folder', type=str, required = True, 
                        help='Folder were the analysis is to be stored.')
    parser.add_argument('--uniprot_input', type=str, required = True, 
                        help='Either UniProts IDSs separated by a comma (Ej, O14746,O14842,O14965) or the path to a txt file containing in each line a UniProt ID.')
    
    parser.add_argument('--path2gesamt', type=str, default = '/opt/xtal/ccp4-9/bin/gesamt',
                        help='Path to the Gesamt executable. By default tries to find it at /opt/xtal/ccp4-9/bin/gesamt')
    
    args = parser.parse_args()
    analysis_folder = os.path.join(args.path2analysis_folder, '')
    path2savetsv = analysis_folder + 'my_tsv.tsv.gz'
    path2gesamt = args.path2gesamt
    uniprot_input = args.uniprot_input
    
    if ',' in uniprot_input or isinstance(uniprot_input, str):
        uniprots = [i.strip() for i in uniprot_input.split(',')]
    else:
        uniprots = [i.strip() for i in open(uniprot_input).read().split('\n')]
        
    if not os.path.exists(analysis_folder):
        os.mkdir(analysis_folder)
    
    path2log = analysis_folder + 'dock_log.txt'
    if not os.path.exists(path2log):
        with open(path2log, 'w') as file:
            pass
    
    with open(path2log, 'a') as file_log:
        now = datetime.datetime.now()
        formato_formal = now.strftime('%d/%m/%Y-%H:%M:%H')
        file_log.write(f'----------{formato_formal}----------\n')
            
    if len([i for i in os.listdir(analysis_folder) if 'uniprot2pdb' in i]) == 2:
        path2uniprot2pdb_raw, path2uniprot2pdb = analysis_folder+'uniprot2pdb_raw.csv', analysis_folder+'uniprot2pdb.csv'
    else: 
        path2uniprot2pdb_raw, path2uniprot2pdb = build_updated_uniprotpdb_dataset(path2savetsv)
    
    uniprot2pdb_raw = pd.read_csv(path2uniprot2pdb_raw)
    uniprot2pdb = pd.read_csv(path2uniprot2pdb)
    
    for nnn, u_i in enumerate(uniprots):
        try:
            uniprot_superimposer_clustering(u_i)
        except Exception as e:
            with open(path2log, 'a') as file_log:
                file_log.write(f'{u_i}: {e}')
        print('--------------')
        print(f'{nnn+1}/{len(uniprots)} UNIPROTS DONE')
        print('--------------')
