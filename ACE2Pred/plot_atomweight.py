import os

import pandas as pd
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import PandasTools


def Convert2SDF():
    df = pd.read_csv('data/Indoor_air_pollutants.csv', header=0, index_col=None)
    print(df.columns)
    df['name'] = df['Name']
    PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Name')
    PandasTools.WriteSDF(df,
                         'data/Indoor_air_pollutants.sdf', properties=list(df.columns),
                         molColName='Name', idName='name')


def Plot_atomweight():
    if 'Indoor_air_pollutants.sdf' not in os.listdir(os.getcwd()):
        Convert2SDF()
    mols = [m for m in Chem.SDMolSupplier('data/Indoor_air_pollutants.sdf') if m is not None]
    df = pd.read_csv(f'data/Fragment_contribution_indoor_air_pollutants.csv', header=0, index_col=0)
    smiles = list(df.index)

    for mol in mols:
        smi = mol.GetProp("SMILES")
        name = mol.GetProp("_Name")
        wt = {}
        if smi in smiles:
            print(name,smi)
            for n, atom in enumerate(Chem.rdmolfiles.CanonicalRankAtoms(mol)):
                wt[n] = df.loc[smi, "Contrib"][n]
            fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, wt, colorMap='bwr',
                                                         alpha=0.5, contourLines=2,size=(300, 300))
            fig.savefig(f'data/figure/{name}.png', bbox_inches='tight', dpi = 600)


Plot_atomweight()

