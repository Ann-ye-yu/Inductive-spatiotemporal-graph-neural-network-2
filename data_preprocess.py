import os
import pickle as pkl
import pandas as pd
import gzip
import json
import numpy as np
import random
import scipy.sparse as sp
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm
try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO

def map_data(data):
    uniq = list(set(data))
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}  # 将作者或者电影id映射到一个新的id,new:0~len(uniq)-1
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)
    return data, id_dict, n

def download_dataset(dataset, files, data_dir):
    """ Downloads dataset if files are not present. """

    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)

        print('Downloading %s dataset' % dataset)

        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'raw_data/' + dataset.replace('_', '-')
        elif dataset == 'ml_10m':
            target_dir = 'raw_data/' + 'ml-10M100K'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(BytesIO(request.read())) as zip_ref:
            zip_ref.extractall('raw_data/')

        os.rename(target_dir, data_dir)
def load_data(fname, seed=1234):
    print('Loading dataset', fname)
    data_dir = 'raw_data/' + fname
    if fname in ['musical_instruments','books']:
        path = data_dir + '/'+fname+'_5.json.gz'
        def parse(path):
            g = gzip.open(path, 'rb')
            for l in g:
                yield json.loads(l)

        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')

        # 用户id,商品id,评分，评价时间戳
        main_cols = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}
        data0 = df[main_cols]
        # 删除用户id只出现了一次的行和商品id只出现少于10次的行
        # uid_counts = list(data0['reviewerID'].value_counts())
        # uid_ = list(data0['reviewerID'].value_counts().index)
        # vid_counts = list(data0['asin'].value_counts())
        # vid_ = list(data0['asin'].value_counts().index)
        # drop_index = []
        # for i, j in enumerate(tqdm(vid_counts)):
        #     if j < 10:
        #         index = data0.loc[data0.asin == vid_[i]].index.to_list()
        #         drop_index.extend(index)
        # for i, j in enumerate(tqdm(uid_counts)):
        #     if j < 10:
        #         index = data0.loc[data0.reviewerID == uid_[i]].index.to_list()
        #         drop_index.extend(index)
        # drop_index = list(set(drop_index))
        # data_1 = data0.drop(index=drop_index)
        data_ = data0.groupby("reviewerID").filter(lambda x: len(x) > 10)
        data_1 = data_.groupby("asin").filter(lambda x: len(x) > 10)
        data_array = data_1.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)
        u_nodes = data_array[:, 0]
        v_nodes = data_array[:, 1]
        ratings = data_array[:, 2].astype(dtypes['ratings'])
        timestamp = data_array[:, 3].astype(dtypes['timestamp'])
        u_nodes_map, u_dict, num_users = map_data(u_nodes)
        v_nodes_map, v_dict, num_items = map_data(v_nodes)
        u_nodes, v_nodes = u_nodes_map.astype(dtypes['u_nodes']), v_nodes_map.astype(dtypes['v_nodes'])
        ratings = ratings.astype(dtypes['ratings'])
        timestamp = timestamp.astype(dtypes['timestamp'])
    else:
        # Check if files exist and download otherwise
        files = ['/ratings.dat', '/movies.dat', '/users.dat']
        download_dataset(fname, files, data_dir)

        sep = r'\:\:'

        filename = data_dir + files[0]
        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

        # use engine='python' to ignore warning about switching to python backend when using regexp for sep
        data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)
        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])
        timestamp = data_array[:, 3].astype(dtypes['timestamp'])
        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
        u_nodes, v_nodes = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)
        timestamp = timestamp.astype(np.float64)
    return num_users, num_items, u_nodes, v_nodes, ratings,  timestamp

def create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None,
                              datasplit_from_file=False, verbose=True, rating_map=None,
                              post_rating_map=None, ratio=1.0):
    """
    Splits data set into train/val/test sets from full bipartite adjacency matrix. Shuffling of dataset is done in
    load_data function.
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix.
    """
    if datasplit_from_file and os.path.isfile(datasplit_path):
        print('Reading processed dataset from file...')
        with open(datasplit_path,'rb') as f:
            num_users, num_items, u_nodes, v_nodes, ratings, timestamp = pkl.load(f)
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    else:
        num_users, num_items, u_nodes, v_nodes, ratings, timestamp = load_data(dataset,seed=seed)
        with open(datasplit_path, 'wb') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, timestamp], f)

    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}
    print("Using random dataset split ...")
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    num_train = ratings.shape[0] - num_val - num_test
    pairs_nonzero = np.vstack([u_nodes, v_nodes]).transpose()

    train_pairs_idx = pairs_nonzero[0:int(num_train * ratio)]
    val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    # test_pair_idx :(100021,2)
    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    all_labels = np.array([rating_dict[r] for r in ratings], dtype=np.int32)
    train_labels = all_labels[0:int(num_train * ratio)]
    val_labels = all_labels[num_train:num_train + num_val]
    test_labels = all_labels[num_train + num_val:]

    train_timestamp = timestamp[0:int(num_train * ratio)]
    val_timestamp = timestamp[num_train:num_train + num_val]
    test_timestamp = timestamp[num_train + num_val:]
    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        timestamp = np.hstack([train_timestamp, val_timestamp])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    if post_rating_map is None:
        data = train_labels + 1.
    else:
        data = np.array([post_rating_map[r] for r in class_values[train_labels]]) + 1.
    data = data.astype(np.float32)
    # u_train_idx: 0-6039
    # v_train_idx: 0-3705
    rating_mx_train = sp.csr_matrix((data, [u_train_idx, v_train_idx]),
                                    shape=[num_users, num_items], dtype=np.float32)
    timestamp_mx_train = sp.csr_matrix((timestamp, [u_train_idx, v_train_idx]),
                                       shape=[num_users, num_items], dtype=np.float64)

    return rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, timestamp_mx_train, num_users, num_items



