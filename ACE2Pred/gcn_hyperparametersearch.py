from deepchem.metrics import Metric, roc_auc_score
from deepchem.models.optimizers import AdaGrad
from deepchem.models.graph_models import GraphConvModel
from deepchem.utils import save_dataset_to_disk, load_dataset_from_disk
import os
import shutil
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def generate_dataset():
    import deepchem as dc
    loader = dc.data.CSVLoader(tasks=['label_ACE2'], feature_field='canonical_smiles', featurizer=dc.feat.ConvMolFeaturizer())
    train_dataset = loader.create_dataset('data/train_oversampling.csv')
    valid_dataset = loader.create_dataset('data/valid.csv')
    test_dataset = loader.create_dataset('data/test.csv')
    save_dataset_to_disk("data/", train_dataset, valid_dataset, test_dataset, [])



def BP_Search(weight_decay, batch_size, dropout, learning_rate):
    if os.path.isdir(dir_name):
        model_dir = f'{dir_name}/wd{round(weight_decay,3)}_bs{round(batch_size,3)}_do{round(dropout,3)}_lr{round(learning_rate,3)}'

        # Training the model
        model = GraphConvModel(n_tasks = 1,
                               batch_size = int(batch_size),
                               dropout = dropout,
                               graph_conv_layer = conv_layer,
                               weight_decay_penalty = 10**-weight_decay,
                               mode = 'classification', 
                               optimizer = AdaGrad(10**-learning_rate),
                               model_dir = model_dir) 
        best_param ={}
        best_param["train_epoch"] = 0
        best_param["valid_epoch"] = 0
        best_param["train_auc"] = 0
        best_param["valid_auc"] = 0
        best_param["test_auc"] = 0

        f_epoch = open(f'{model_dir}/epoch_log.csv','w')
        f_epoch.write('epoch,train_loss,train_auc,valid_auc,test_auc\n')

        for epoch in range(500):
            # Fit trained model
            model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)
            train_auc = model.evaluate(train_dataset, [metric], [])['roc_auc_score']
            valid_auc = model.evaluate(valid_dataset, [metric], [])['roc_auc_score']

            print(f'{model_dir}--epoch:{epoch}, train_auc:{train_auc}, valid_auc:{valid_auc}')
            
            # save the result of each epoch 
            f_epoch.write(','.join([str(epoch),str('%.4f'%train_auc),str('%.4f'%valid_auc)])+'\n')
            
            # Compare to the best result of previous epochs
            if train_auc > best_param["train_auc"]:
                best_param["train_epoch"] = epoch
                best_param["train_auc"] = train_auc
            if valid_auc > best_param["valid_auc"]:
                best_param["valid_epoch"] = epoch
                best_param["valid_auc"] = valid_auc
                model.save_checkpoint(max_checkpoints_to_keep=1,model_dir=f'{model_dir}')

            # early stopping
            if (epoch - best_param["train_epoch"] > 6) and (epoch - best_param["valid_epoch"] > 8):
                f_epoch.close()
                break

        print(f'{model_dir}--best epoch:{best_param["valid_epoch"]}, train_auc:{train_auc}, valid_auc:{valid_auc}, test_auc:{test_auc}')   

        # write the best result
        with open(log_file,'a') as f:
            f.write(','.join([str(weight_decay), str(batch_size), str(dropout), str(learning_rate), 
                        str(best_param["valid_epoch"]), str(best_param["train_auc"]), 
                        str(best_param["valid_auc"])]) + '\n')

        return best_param["valid_auc"]



# generate_dataset()
_, datasets, _ = load_dataset_from_disk("data/")
train_dataset, valid_dataset, test_dataset = datasets
metric = Metric(roc_auc_score, mode="classification")


# clear out existing logs before every run
conv_layers = [[32, 32, 64, 64, 128, 128],[32, 32, 64, 64, 128, 128, 256, 256]]
base_dir_name = 'hyperparameter_search/'  # the default folder saves the data for [64, 64, 128, 128]
for conv_layer in conv_layers:
    print(conv_layer)
    dir_name = base_dir_name +str(conv_layer)
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

    log_file = dir_name + '/GCN_BayesionOptimazation.log'
    with open(log_file,'w') as f:
        f.write(','.join(['weight_decay', 'batch_size','dropout', 
                'learning_rate', 'best_valid_epoch', 'train_auc', 'valid_auc'])+'\n')

    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')

    model_params = {
        'weight_decay':('cont', [2, 6]),
        'batch_size': ('int', [64, 256]),
        'dropout':  ('cont', [0, 0.5]),
        'learning_rate': ('cont', [2, 5])}

    gpgo = GPGO(gp, acq, BP_Search, model_params)
    gpgo.run(max_iter=50, init_evals=2)

