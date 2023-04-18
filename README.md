# MGFmiRNAloc
## Data

#### SMILES strings for RNA bases

```shell
cat ./data/basesSMILES/ATGC.dict
```


```

**Positive.txt** is the input file of positive samples. **Negative.txt** is the input file of negative samples. **./train_val_test_10CV_data/*** is the folder including training, validation, and test set of each fold.

```

**pos.txt** is the input file of positive samples. **neg.txt** is the input file of negative samples. **./train_val_test_10CV_data/*** is the folder including training, validation, and test set of each fold.



The code has been tested running under Python 3.5.6. 

The required packages are as follows:

* cudatoolkit=10.0
* cudnn=7.6.4
* tensorflow-gpu==1.13.2
* keras==2.1.5
* numpy==1.16.4
* pandas==0.20.3
* scikit-learn==0.22.2.post1
* rdkit==2017.03.1



# example for train
python ./code/main.py -p ./data/Positive.txt -n ./data/Negative.txt -f 10 -o ./result/
