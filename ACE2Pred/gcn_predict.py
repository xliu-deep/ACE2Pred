
import os
import pickle

import deepchem as dc
import pandas as pd
from deepchem.models.graph_models import GraphConvModel
from deepchem.models.optimizers import AdaGrad


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def GCN_model_external(model):
    loader = dc.data.CSVLoader(tasks=['ACE2'], feature_field='SMILES', featurizer=dc.feat.ConvMolFeaturizer())
    external_dataset = loader.create_dataset('data/Indoor_air_pollutants.csv')
    external_smiles = pd.read_csv('data/Indoor_air_pollutants.csv',header = 0,index_col = None)

    external_predlabel = model.predict(external_dataset)[:,0][:,1]
    print(external_predlabel.shape)
    external_smiles['Pred_label'] = external_predlabel
    external_smiles.to_csv(r'data\Indoor_air_pollutants_Predict_result.csv')


def Fragmentation():
    dataset = ['data/Indoor_air_pollutants.csv','Indoor_prepare','SMILES']
    # dataset = ['../L1000/L1000Tox_Combined.csv', 'L1000Tox_Combined','canonical_smiles']
    loader = dc.data.CSVLoader(tasks=[],
                               feature_field=dataset[2],
                               featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True))
    frag_dataset = loader.create_dataset(dataset[0], shard_size=5000)
    print(frag_dataset.X.shape)
    tr = dc.trans.FlatteningTransformer(frag_dataset)
    frag_dataset = tr.transform(frag_dataset)
    print(frag_dataset.X.shape)
    with open(f'data/Indoor_air_pollutants_frag.pickle','wb') as f:
        pickle.dump(frag_dataset,f)


def Fragment_contribution(model):
    loader = dc.data.CSVLoader(tasks=[], feature_field='SMILES', featurizer=dc.feat.ConvMolFeaturizer())
    dataset = loader.create_dataset('data/Indoor_air_pollutants.csv')

    pred = model.predict(dataset)[:, 0][:, 1]
    pred = pd.DataFrame(pred, index=dataset.ids, columns=["Molecule"])  # turn to dataframe for convenience

    # fragments
    Fragmentation()
    with open(f'data/Indoor_air_pollutants_frag.pickle', 'rb') as f:
        frag_dataset = pickle.load(f)
    pred_frags = model.predict(frag_dataset)[:, 0][:, 1]
    pred_frags = pd.DataFrame(pred_frags, index=frag_dataset.ids,
                              columns=["Fragment"])  # turn to dataframe for convenience

    # merge 2 dataframes by molecule names
    df = pd.merge(pred_frags, pred, right_index=True, left_index=True)
    # find contribs
    df['Contrib'] = df["Molecule"] - df["Fragment"]
    df.to_csv(f'data/Fragment_contribution_indoor_air_pollutants.csv')



model_dir = r'GCN_final_model'
model_params = {
    'n_tasks':1,
    'weight_decay_penalty':10**-5.216935,
    'batch_size': int(83.2398256),
    'dropout': 0.083337,
    'mode':'classification',
    'optimizer': AdaGrad(10**-2.0),
    'graph_conv_layer': [32, 32, 64, 64, 128, 128],
    'model_dir': model_dir}


def main():
    model = GraphConvModel(**model_params)
    model.restore()
    # GCN_model_external(model)
    Fragment_contribution(model)


if __name__ == '__main__':
    main()


