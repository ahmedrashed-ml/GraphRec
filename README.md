# GraphRec
This is our implementation for the recsys 2019 paper:

Ahmed Rashed, Josif Grabocka, and Lars Schmidt-Thieme. 2019. Attribute-Aware Non-Linear Co-Embeddings of Graph Features . InThirteenth ACMConference on Recommender Systems (RecSys ’19)
## Enviroment 
	* pandas==1.0.3
	* tensorflow==1.14.0
	* matplotlib==3.1.3
	* numpy==1.18.1
	* six==1.14.0
	* scikit_learn==0.23.1
  
## Steps
1. Just uncomment the respective code of the dataset you want to reproduce the results for and run "python GraphRec.py".

## Paper
Preprint version : https://datasink.ismll.uni-hildesheim.de/oc/index.php/s/S432p92Le3HZscd

## Supplementary/Extra Results
### ML100k Experiment using the u1.base/u1.test splits

Model | RMSE
------------ | -------------
GraphRec (w/ Graph Feat.)  | 0.904
GraphRec (w/ Graph Feat. & Users/Items Attributes)  | 0.897
