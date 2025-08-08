### Dependency
Python 3.8, 
Pytorch 1.9.1,
scipy==1.10.0
scikit-learn==1.3.1
tqdm==4.66.1
### Dataset
we evaluate our AHCFCDR framework using the widely-used Amazon review dataset: https://jmcauley.ucsd.edu/data/amazon/.

You can make a directory ./data in the root and download the data into it.
    
### Model Training
Our model needs a two-stage training. 

**First**, you need to run the follow script to pretraing.
    
    nohup python -u pretrain.py --test_ratio 0.2 --dataset 1 >> sh.log>&1 &

**Second**, run this script to traning.

    nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 3 --emb_size 64 --hid_size 128 >> sh_train.log>&1 &

If you want to see the training process, you can use the command： 
    
    tail -f sh_train.log

### Test
Test the model by this command.

    nohup python -u test.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 3 --emb_size 64 --hid_size 128 >> sh_test.log>&1 &

If you want to see the test process, you can use the command：
    
    tail -f sh_test.log

### Parameter Configuration:

- `--test_ratio`: test ratio within `0.2,0.5,0.8`, default for `0.2`

- `--dataset *`: You can run our method on this three dataset by setting the follow parameter.
    *parameter`--dataset 1`represents dataset `tgt_CDs_and_Vinyl_src_Movies_and_TV`,

    *parameter`--dataset 2` represents dataset `tgt_Movies_and_TV_src_Books`

    *parameter`--dataset 3` represents dataset `tgt_CDs_and_Vinyl_src_Books`. 
- `--head_num`: the number of heads of attention, default for 3.
- `--emb_size`: representational dimension, default for 64.
- `--hid_size`: hidden layer dimension, default for 128.
