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
import h5py
from tqdm import tqdm
from collections import Counter

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
    if fname in ['musical_instruments', 'books']:
        path = data_dir + '/' + fname + '_5.json.gz'

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
    return num_users, num_items, u_nodes, v_nodes, ratings, timestamp


def create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None,
                              datasplit_from_file=False, verbose=True, rating_map=None,
                              post_rating_map=None, ratio=1.0):
    """
    Splits data set into train/val/test sets from full bipartite adjacency matrix. Shuffling of dataset is done in
    load_data function.
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix.
    """
    # if datasplit_from_file and os.path.isfile(datasplit_path):
    #     print('Reading processed dataset from file...')
    #     with open(datasplit_path, 'rb') as f:
    #         num_users, num_items, u_nodes, v_nodes, ratings, timestamp = pkl.load(f)
    #     print('Number of users = %d' % num_users)
    #     print('Number of items = %d' % num_items)
    #     print('Number of links = %d' % ratings.shape[0])
    #     print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))
    #
    # else:
    num_users, num_items, u_nodes, v_nodes, ratings, timestamp = load_data(dataset, seed=seed)
    M = np.zeros((6040, 3706))
    ratings_index = 0
    # row = u_nodes_ratings[0]-1
    # col = v_nodes_ratings[0]-1
    for i in range(len(ratings)):
        row = u_nodes[i]
        col = v_nodes[i]
        M[row][col] = ratings[i]
    count_dict = {
        '<=10': 0,
        '(10,20]': 0,
        '(20,30]': 0,
        '(30,40]': 0,
        '(40,50]': 0,
        '(50,100]': 0,
        '>100': 0
    }
    for row in range(len(M)):
        # print(np.count_nonzero(M[row]))
        count_row = np.count_nonzero(M[row])
        if count_row <= 10:
            count_dict["<=10"] += 1
        elif 10 < count_row <= 20:
            count_dict["(10,20]"] += 1
        elif 20 < count_row <= 30:
            count_dict["(20,30]"] += 1
        elif 30 < count_row <= 40:
            count_dict["(30,40]"] += 1
        elif 40 < count_row <= 50:
            count_dict["(40,50]"] += 1
        elif 50 < count_row <= 100:
            count_dict["(50,100]"] += 1
        elif count_row > 100:
            count_dict[">100"] += 1


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


def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    test1 = db.keys()
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


def load_data_monti(dataset, testing=False, rating_map=None, post_rating_map=None):
    """
    Loads data from Monti et al. paper.
    if rating_map is given, apply this map to the original rating matrix
    if post_rating_map is given, apply this map to the processed rating_mx_train without affecting the labels
    """

    path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat'

    M = load_matlab_file(path_dataset, 'M')
    if rating_map is not None:
        M[np.where(M)] = [rating_map[x] for x in M[np.where(M)]]

    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')

    num_users = M.shape[0]
    num_items = M.shape[1]

    if dataset == 'flixster':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        Wcol = load_matlab_file(path_dataset, 'W_movies')
        u_features = Wrow
        v_features = Wcol
    elif dataset == 'douban':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        u_features = Wrow
        v_features = np.eye(num_items)
    elif dataset == 'yahoo_music':
        Wcol = load_matlab_file(path_dataset, 'W_tracks')
        u_features = np.eye(num_users)
        v_features = Wcol

    u_nodes_ratings = np.where(M)[0]
    v_nodes_ratings = np.where(M)[1]
    ratings = M[np.where(M)]

    M_ = np.zeros((3000, 3000))
    ratings_index = 0
    # row = u_nodes_ratings[0]-1
    # col = v_nodes_ratings[0]-1
    for i in range(len(ratings)):
        row = u_nodes_ratings[i]
        col = v_nodes_ratings[i]
        M_[row][col] = ratings[i]

    count_dict = {
        '(0,10]': 0,
        '(10,20]': 0,
        '(20,30]': 0,
        '(30,40]': 0,
        '(40,50]': 0,
        '(50,100]': 0,
        '>100': 0
    }
    for row in range(len(M_)):
        # print(np.count_nonzero(M[row]))
        count_row = np.count_nonzero(M_[row])
        if 0<count_row <= 10:
            count_dict["(0,10]"] += 1
        elif 10 < count_row <= 20:
            count_dict["(10,20]"] += 1
        elif 20 < count_row <= 30:
            count_dict["(20,30]"] += 1
        elif 30 < count_row <= 40:
            count_dict["(30,40]"] += 1
        elif 40 < count_row <= 50:
            count_dict["(40,50]"] += 1
        elif 50 < count_row <= 100:
            count_dict["(50,100]"] += 1
        elif count_row > 100:
            count_dict[">100"] += 1

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print('number of users = ', len(set(u_nodes)))
    print('number of item = ', len(set(v_nodes)))

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges

    num_train = np.where(Otraining)[0].shape[0]
    num_test = np.where(Otest)[0].shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero_train = np.array([[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
    idx_nonzero_train = np.array([u * num_items + v for u, v in pairs_nonzero_train])

    pairs_nonzero_test = np.array([[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    idx_nonzero_test = np.array([u * num_items + v for u, v in pairs_nonzero_test])

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert (len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    '''Note here rating matrix elements' values + 1 !!!'''
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.

    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    if u_features is not None:
        u_features = sp.csr_matrix(u_features)
        print("User features shape: " + str(u_features.shape))

    if v_features is not None:
        v_features = sp.csr_matrix(v_features)
        print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values


def load_official_trainvaltest_split(dataset, testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    """
    Loads official train/test split and uses 10% of training samples for validaiton
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
    """

    sep = '\t'

    # Check if files exist and download otherwise
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    fname = dataset
    data_dir = 'raw_data/' + fname

    download_dataset(fname, files, data_dir)

    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}

    filename_train = 'raw_data/' + dataset + '/u1.base'
    filename_test = 'raw_data/' + dataset + '/u1.test'

    data_train = pd.read_csv(
        filename_train, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_test = pd.read_csv(
        filename_test, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)

    if ratio < 1.0:
        data_array_train = data_array_train[data_array_train[:, -1].argsort()[:int(ratio * len(data_array_train))]]

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    timestamp = data_array[:, 3].astype(dtypes['timestamp'])
    M = np.zeros((943,1682))
    ratings_index = 0
    # row = u_nodes_ratings[0]-1
    # col = v_nodes_ratings[0]-1
    for i in range(len(ratings)):
        row = u_nodes_ratings[i]-1
        col = v_nodes_ratings[i]-1
        M[row][col] = ratings[i]
    count_dict = {
        '<=10': 0,
        '(10,20]': 0,
        '(20,30]': 0,
        '(30,40]': 0,
        '(40,50]': 0,
        '(50,100]': 0,
        '>100': 0
    }
    for row in range(len(M)):
        # print(np.count_nonzero(M[row]))
        count_row = np.count_nonzero(M[row])
        if count_row <= 10:
            count_dict["<=10"] += 1
        elif 10 < count_row <= 20:
            count_dict["(10,20]"] += 1
        elif 20 < count_row <= 30:
            count_dict["(20,30]"] += 1
        elif 30 < count_row <= 40:
            count_dict["(30,40]"] += 1
        elif 40 < count_row <= 50:
            count_dict["(40,50]"] += 1
        elif 50 < count_row <= 100:
            count_dict["(50,100]"] += 1
        elif count_row > 100:
            count_dict[">100"] += 1
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)
    timestamp = timestamp.astype(np.float64)
    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    time_labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    time_labels[u_nodes, v_nodes] = np.array(timestamp)

    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])
        assert (time_labels[u_nodes[i], v_nodes[i]] == timestamp[i])

    labels = labels.reshape([-1])
    time_labels = time_labels.reshape([-1])
    # number of test and validation edges, see cf-nade code

    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert (labels[idx_nonzero[i]] == rating_dict[ratings[i]])
        assert (time_labels[idx_nonzero[i]] == timestamp[i])

    idx_nonzero_train = idx_nonzero[0:num_train + num_val]
    idx_nonzero_test = idx_nonzero[num_train + num_val:]

    pairs_nonzero_train = pairs_nonzero[0:num_train + num_val]
    pairs_nonzero_test = pairs_nonzero[num_train + num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert (len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    train_time_labels = time_labels[train_idx]
    val_time_labels = time_labels[val_idx]
    test_time_labels = time_labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    timestamp_mx_train = np.zeros(num_users * num_items, dtype=np.float64)
    timestamp_mx_train[train_idx] = time_labels[train_idx].astype(np.float64)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.

    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))
    timestamp_mx_train = sp.csr_matrix(timestamp_mx_train.reshape(num_users, num_items))

    if dataset == 'ml_100k':

        # movie features (genres)
        sep = r'|'
        movie_file = 'raw_data/' + dataset + '/u.item'
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python', encoding='ISO-8859-1')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # user features

        sep = r'|'
        users_file = 'raw_data/' + dataset + '/u.user'
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        age = users_df['age'].values
        age_max = age.max()

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age'] / np.float(age_max)
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

    elif dataset == 'ml_1m':

        # load movie features
        movies_file = 'raw_data/' + dataset + '/movies.dat'

        movies_headers = ['movie_id', 'title', 'genre']
        movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

        # extracting all genres
        genres = []
        for s in movies_df['genre'].values:
            genres.extend(s.split('|'))

        genres = list(set(genres))
        num_genres = len(genres)

        genres_dict = {g: idx for idx, g in enumerate(genres)}

        # creating 0 or 1 valued features for all genres
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                gen = s.split('|')
                for g in gen:
                    v_features[v_dict[movie_id], genres_dict[g]] = 1.

        # load user features
        users_file = 'raw_data/' + dataset + '/users.dat'
        users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        # extracting all features
        cols = users_df.columns.values[1:]

        cntr = 0
        feat_dicts = []
        for header in cols:
            d = dict()
            feats = np.unique(users_df[header].values).tolist()
            d.update({f: i for i, f in enumerate(feats, start=cntr)})
            feat_dicts.append(d)
            cntr += len(d)

        num_feats = sum(len(d) for d in feat_dicts)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                for k, header in enumerate(cols):
                    u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.
    else:
        raise ValueError('Invalid dataset option %s' % dataset)

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    print("User features shape: " + str(u_features.shape))
    print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, timestamp_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, num_users, num_items
