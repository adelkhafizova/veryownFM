import numpy as np
import pandas as pd

import gc
import os

from tqdm import tqdm_notebook, tqdm
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, mean_squared_error


dtypes = {
    'user':   'uint32',
    'movie':  'uint32',
    'rating': 'float16'
}

# построение различных признаков для movielens
def build_movielens(folder, test_size, with_genres=True, with_users_info=True, with_rated_movies=True):
    print('load ratings....')
    rating_path = [name for name in os.listdir(folder) if 'ratings' in name][0]
    if 'csv' in rating_path:
        ratings = pd.read_csv(folder + rating_path, sep=',', header=0, dtype=dtypes,
                      names=['user', 'movie', 'rating', 'timestamp'])
        ratings['rating'] = (ratings['rating'] + 0.5).astype('int8')
    else:
        ratings = pd.read_csv(folder + rating_path, sep='::', header=None, engine='python',
                              names=['user', 'movie', 'rating', 'timestamp'], dtype=dtypes)
    ratings.rating = ratings.rating.astype('int8')
    
    print('calculation of monthes....')
    ratings['timestamp'] = pd.to_datetime(ratings.timestamp, unit='s')
    min_date = ratings.timestamp.min()
    ratings['monthes'] = (ratings.timestamp - min_date).dt.days // 28
    ratings.monthes /= ratings.monthes.max()
    ratings.monthes = ratings.monthes.astype('float16')
    dataset = ratings.drop('timestamp', 1)
    del(ratings); gc.collect()
    
    if with_genres:
        print('load movies....')
        movie_path = [name for name in os.listdir(folder) if 'movies' in name][0]
        if 'csv' in movie_path:
            movies = pd.read_csv(folder + movie_path, sep=',', header=0,
                                 names=['movie', 'title', 'genres'], 
                                 usecols=['movie', 'genres'], dtype=dtypes)
        else:
            movies = pd.read_csv(folder + movie_path, sep='::', engine='python',
                                 names=['movie', 'title', 'genres'], 
                                 usecols=['movie', 'genres'], header=None, dtype=dtypes)

        print('build genres ohe....')
        sparse_genres = CountVectorizer().fit_transform(movies.genres.map(lambda x: x.replace('|', ' ')))
        colnames = ['genre_{}'.format(col) for col in range(sparse_genres.shape[1])]
        sparse_genres = pd.DataFrame(sparse_genres.todense().astype('uint8'), columns=colnames)
        movies = pd.concat([movies[['movie']], sparse_genres], axis=1)
        del(sparse_genres); gc.collect()        

        print('join dataframes....')
        dataset = pd.merge(dataset, movies, on='movie', how='inner')
        del(movies); gc.collect()
    else:
        print('genres skipped')
    
    if with_users_info and 'users.dat' in os.listdir(folder):
        print('load users info....')
        users = pd.read_csv(folder + 'users.dat', sep='::', 
                            header=None, names=['user', 'gender', 'age', 'occupation', 'zip'],
                            engine='python')
        users.age    = (users.age / users.age.max()).astype('float16')
        users.gender = users.gender.apply(lambda x: 1 if x=='M' else 0).astype('int8')
        users.occupation = users.occupation.astype('int8')
        users.zip    = np.unique(users.zip.values, return_inverse=True)[1]
        users.zip = users.zip.astype('int16')
        dataset = pd.merge(dataset, users, on='user', how='left')
        del(users); gc.collect()
    else:
        print('users info skipped')

    np.random.seed(42)
    print('train/test split...')
    test_indexes = np.random.choice(dataset.index, int(test_size * dataset.shape[0]), replace=False)
    test = dataset.loc[test_indexes]
    train = dataset.drop(test_indexes)
    del(dataset); gc.collect();
    
    if with_rated_movies:
        print('building rated movies history (on train)....')
        rated_movies = train.groupby('user')['movie'].agg(lambda x: list(x))
        train.loc[:, 'ratedMovies'] = train.user.map(rated_movies)
        test.loc[:, 'ratedMovies']  = test.user.map(rated_movies)
        del(rated_movies); gc.collect()
    else:
        print('rated movies history skipped')
        
    print('preprocessing done....')
    return train, test


# хелпер для кодирования категориальных признаков
def get_offset_stats(train, test):
    offset_stats = {}
    offset_stats['users_len']  = train.user.append(test.user).max() + 1#6040
    offset_stats['movie_len'] = train.movie.append(test.movie).max() + 1 #3952
    offset_stats['genre_len']  = len([col for col in train.columns if 'genre' in col])
    if 'occupation' in train.columns:
        offset_stats['occupation_len'] = train.occupation.append(test.occupation).max() + 1
        offset_stats['zip_len']        = train.zip.append(test.zip).max() + 1
    return offset_stats


# функции, преобразующие датасет в формат, заданный feature_extractor
def train2format(data, features_extractor, offset_lens, train='train',
                 with_normalization=False,
                 with_user_features=False,
                 with_rated_films=False):
    
    writer_train      = open(train, 'w')    
    for row in tqdm(data.iterrows(), total=data.shape[0], miniters=1000):
        features = features_extractor(
            row[1], offset_lens, with_normalization,
            with_user_features, with_rated_films)
        
        label = str(int(row[1]['rating']))
        output_line = '{0} {1}\n'.format(label, features)
        writer_train.write(output_line)            
    writer_train.close()

def test2format(data, features_extractor, offset_lens, 
                x_test_output='test', y_test_output='ytest', 
                with_normalization=False,
                with_user_features=False,
                with_rated_films=False):
    
    writer_test = open(x_test_output, 'w')
    writer_ytest = open(y_test_output, 'w')
    for row in tqdm(data.iterrows(), total=data.shape[0]):
        label = str(int(row[1]['rating']))
        features = features_extractor(
            row[1], offset_lens, with_normalization,
            with_user_features, with_rated_films)
        
        output_line = '{0} {1}\n'.format(label, features)
        writer_test.write(output_line)
        writer_ytest.write('%s\n' % label) 
    
    writer_test.close()
    writer_ytest.close()
    
    
def fm_extractor(row, field_info, with_normalization=False, with_user_features=False, with_rated_films=False):
    offset = 0
    current_cat_value = ('{0:.2}'.format(1 / field_info['users_len']) if with_normalization else '1')
    output_line = '{0}:{1} '.format(int(row['user']) + offset, current_cat_value)
    
    offset += field_info['users_len']
    current_cat_value = ('{0:.2}'.format(1 / field_info['movie_len']) if with_normalization else '1')
    output_line += '{0}:{1} '.format(int(row['movie']) + offset, current_cat_value)
    
    offset += field_info['movie_len']
    output_line += '{0}:{1:.1} '.format(offset, row['monthes'])
    
    offset += 1
    current_cat_value = ('{0:.2}'.format(1 / field_info['genre_len']) if with_normalization else '1')
    for genre_index in range(field_info['genre_len']):
        if row['genre_{}'.format(genre_index)] == 1:
            output_line += '{0}:{1} '.format(offset + genre_index, current_cat_value)
            
    offset += field_info['genre_len']
    if with_user_features:
        output_line += '{0}:{1} '.format(offset, row['gender'])
        offset += 1
        output_line += '{0}:{1:.1} '.format(offset, row['age'])
        offset += 1
        
        current_cat_value = ('{0:.2}'.format(1 / field_info['occupation_len']) if with_normalization else '1')
        output_line += '{0}:{1} '.format(row['occupation'] + offset, current_cat_value)
        offset += field_info['occupation_len']
        
        current_cat_value = ('{0:.2}'.format(1 / field_info['zip_len']) if with_normalization else '1')
        output_line += '{0}:{1} '.format(row['zip'] + offset, current_cat_value)
        offset += field_info['zip_len']
    
    if with_rated_films:
        n_rated_movies = len(row['ratedMovies'])
        for movie_id in row['ratedMovies']:
            output_line += '{0}:{1:.3} '.format(movie_id + offset, 1 / n_rated_movies)
        
    return output_line


### Сonversion between vw/fm and regression/classification
def fm2vw(infile, outfile, split_pos=1):    
    input_file = open(infile,  'r')
    out_file   = open(outfile, 'w')
    for line in tqdm(input_file):
        out_file.write(line[:split_pos] + ' |' + line[split_pos:])

def reg2fm(infile, outfile):    
    input_file = open(infile,  'r')
    out_file   = open(outfile, 'w')
    for line in tqdm(input_file):
        target = str(int(int(line[0]) > 3))
        out_file.write(target + line[1:])

def reg2vw(infile, outfile):    
    input_file = open(infile,  'r')
    out_file   = open(outfile, 'w')
    for line in tqdm(input_file):
        target = str((int(line[0]) > 3) * 2 - 1)
        out_file.write(target + ' |' + line[1:])
        
def target_transform(infile, outfile):    
    input_file = open(infile,  'r')
    out_file   = open(outfile, 'w')
    for line in tqdm(input_file):
        sym_shift = 2 if line[1] == '0' else 1  
        target = str((int(line[:sym_shift]) + 1) / 2) + ' '
        out_file.write(target + line[sym_shift:])     
    
    
### Функции, позволяющие оценить те или иные метрики, не загружаю в память таблички
def get_rmse(ytest_input='ytest', pred_input='pred'):
    '''inplace rmse'''
    n, loss = 0, 0
    reader_ytest = open(ytest_input, 'r')
    reader_pred = open(pred_input, 'r')

    for label, pred in zip(reader_ytest, reader_pred):    
        n+=1
        true_score = float(label)
        pred_score = float(pred)
        loss += np.square(pred_score - true_score)
    reader_ytest.close()
    reader_pred.close()
    return np.sqrt(loss / n)


def get_log_loss(ytest_input='ytest', pred_input='pred'):
    '''inplace logloss'''
    n, loss = 0, 0
    reader_ytest = open(ytest_input, 'r')
    reader_pred = open(pred_input, 'r')

    for label, pred in zip(reader_ytest, reader_pred):    
        n+=1        
        true_label = int(label)
        pred_score = float(pred)
        pred_score = 1 / (1 + np.exp(-pred_score))
        loss -= true_label * np.log(pred_score) + (1 - true_label) * np.log(1 - pred_score) 
        
    reader_ytest.close()
    reader_pred.close()
    return loss / n

