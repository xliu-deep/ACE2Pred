from rdkit import Chem
from molvs import Standardizer
from molvs.fragment import LargestFragmentChooser
from molvs.charge import Uncharger
from molvs.tautomer import TautomerCanonicalizer


def Standardize(mol):
    mol = Standardizer().standardize(mol)
    mol = LargestFragmentChooser()(mol)
    mol = Uncharger()(mol)
    mol = TautomerCanonicalizer().canonicalize(mol)
    smiles = Chem.MolToSmiles(mol)

    return smiles


def main():
    # The example of input smiles file:
    # canonical_smiles	pert_id
    # C#C[C@]1(O)CCC2C3CC=C4CC(=O)CCC4C3CC[C@@]21C	BRD-A00758722
    # COc1cc(C(O)C(C)N)ccc1O	BRD-A00763758
    # Cn1c(=O)c2c(ncn2CC(O)CO)n(C)c1=O	BRD-A00827783
    # C=CCc1ccccc1OCC(O)CNC(C)C	BRD-A00993607

    fin = open(r'input.smiles', 'r', encoding='utf-8')
    fout = open(r'output_standardized.smiles', 'w', encoding='utf-8')
    lines = fin.readlines()
    fout.write(lines[0])

    for line in lines[1:]:
        items = line.strip().split('\t')
        smiles = items[0]
        mol = Chem.MolFromSmiles(smiles)
        newsmiles = Standardize(mol)
        newline = [newsmiles] + items[1:]
        fout.write('\t'.join(newline) + '\n')
    fin.close()
    fout.close()


if __name__ == '__main__':
    main()
