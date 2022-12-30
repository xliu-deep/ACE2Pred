![image](https://user-images.githubusercontent.com/1555415/210035930-d73a4101-79ff-4a1a-a29c-e097bc30a3df.png)
# ACE2Pred

Despite the fact that coronavirus disease 2019 (COVID-19), caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), has been disrupting human life and health worldwide for several years, the interaction between exogenous substances and pathogens is not well understood. During viral infection, organism receptors play a significant role in mediating the entry of viruses into host cells. Taking angiotensin-converting enzyme 2 (ACE2), the major receptor of SARS-CoV-2, as an example, an interpretable deep learning model based on graph convolutional neural network (GCN) was constructed for the first time to predict how exogenous substances affect the transcriptional expression of the ACE2 gene in the study. Systematic analysis showed that the proposed GCN model outperformed other machine learning models, and the area under receiver operating characteristic curve (AUROC) reached 0.712 and 0.703 on validation set and test set, respectively. In addition, qPCR experiment confirmed the predicted results of ACE2 gene upregulation by indoor air pollutants. Moreover, according to the predicted results, 5.77% of indoor air pollutants are likely to induce ACE2 gene upregulation. The proposed framework is expected to provide a paradigm and methodological foundation for the prediction of the interaction between pathogen and exogenous substances from environmental media. The impact of microorganisms may not be neglected in risk assessment of pollutants.

## This library contains all the data for modeling and testing, as well as the predictive model for exogenous substances-induced ACE2 regulation.


### Molecular GCN architecture for ACE2 gene change prediction
![image](https://user-images.githubusercontent.com/1555415/190976096-3358dd8a-5a1c-4244-a2d6-e963b91549ae.png)

#### Model 
- GCN_final_model.zip
  > ready-to-use GCN model
- gcn_hyperparametersearch.py
  > hyperparameter tunning for GCN model
- gcn_predict.py
  > prediction and evaluation process
- plot_atomweight.py
  > plot the atomic weights of indoor air pollutants that contribute the ACE2 gene upregulation
  > Fragment_contribution_indoor_air_pollutants.csv
#### Data folder





