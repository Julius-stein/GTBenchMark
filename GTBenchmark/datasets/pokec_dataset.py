import re
import os.path as osp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Callable, List, Optional
from sklearn.model_selection import train_test_split
from torch_geometric.data import (
    Data,
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import index_to_mask, to_undirected

import gzip
import shutil
import os
import time


def get_categorical_features(feat_df, cat_cols):
    one_hot_encoded_df = pd.get_dummies(feat_df[cat_cols], columns=cat_cols)
    cat_features = torch.tensor(one_hot_encoded_df.values, dtype=torch.float32)
    
    return cat_features

def get_numerical_features(feat_df, num_cols):
    feat_df[num_cols] = feat_df[num_cols].fillna(0.0)
    num_feats = torch.tensor(feat_df[num_cols].values, dtype=torch.float32)
    
    return num_feats

def normalize(feature_matrix):
    mean = torch.mean(feature_matrix, axis=0)
    stdev = torch.sqrt(torch.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0]) + 1e-9
    return mean, stdev, (feature_matrix - mean) / stdev

class PokecDataset(InMemoryDataset):
    r"""
    H-Pokec is a heterogeneous friendship graph of a Slovalk online social
    network, collected from `SNAP at Stanford University <https://snap.stanford.edu/data>`_.
    
    The dataset consists of multiple types of entities--users and multiple
    fields of the hobby clubs they joined (e.g., movies, music)--as well as
    multiple types of directed relation representing the friendship relations
    and the hobby clubs they joined. Each user node is associated with a
    66-dimensional feature vector extracted from the user profile information,
    such as geographical region, age, and visibility of user profile. Each user
    node is labeled with a binary label tagging their reported gender. This
    dataset is randomly split into training, validation, and test set.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    
    """

    urls = [
        "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz",
        "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"
        ]
    
    node_fields = [
        "public",
        "completion_percentage",
        "gender",
        "region",
        "last_login",
        "registration",
        "AGE",
        "body",
        "I_am_working_in_field",
        "spoken_languages",
        "hobbies",
        "I_most_enjoy_good_food",
        "pets",
        "body_type",
        "my_eyesight",
        "eye_color",
        "hair_color",
        "hair_type",
        "completed_level_of_education",
        "favourite_color",
        "relation_to_smoking",
        "relation_to_alcohol",
        "sign_in_zodiac",
        "on_pokec_i_am_looking_for",
        "love_is_for_me",
        "relation_to_casual_sex",
        "my_partner_should_be",
        "marital_status",
        "children",
        "relation_to_children",
        "I_like_movies",
        "I_like_watching_movie",
        "I_like_music",
        "I_mostly_like_listening_to_music",
        "the_idea_of_good_evening",
        "I_like_specialties_from_kitchen",
        "fun",
        "I_am_going_to_concerts",
        "my_active_sports",
        "my_passive_sports",
        "profession",
        "I_like_books",
        "life_style",
        "music",
        "cars",
        "politics",
        "relationships",
        "art_culture",
        "hobbies_interests",
        "science_technologies",
        "computers_internet",
        "education",
        "sport",
        "movies",
        "travelling",
        "health",
        "companies_brands",
        "more",
        ""
    ]
    
    node_features = ["body",
        "I_am_working_in_field",
        "spoken_languages",
        "hobbies",
        "I_most_enjoy_good_food",
        "pets",
        "body_type",
        "my_eyesight",
        "eye_color",
        "hair_color",
        "hair_type",
        "completed_level_of_education",
        "favourite_color",
        "relation_to_smoking",
        "relation_to_alcohol",
        "sign_in_zodiac",
        "on_pokec_i_am_looking_for",
        "love_is_for_me",
        "relation_to_casual_sex",
        "my_partner_should_be",
        "marital_status",
        "children",
        "relation_to_children",
        "I_like_movies",
        "I_like_watching_movie",
        "I_like_music",
        "I_mostly_like_listening_to_music",
        "the_idea_of_good_evening",
        "I_like_specialties_from_kitchen",
        "fun",
        "I_am_going_to_concerts",
        "my_active_sports",
        "my_passive_sports",
        "profession",
        "I_like_books"
    ]
    
    edge_features = ["life_style",
        "music",
        "cars",
        "politics",
        "relationships",
        "art_culture",
        "hobbies_interests",
        "science_technologies",
        "computers_internet",
        "education",
        "sport",
        "movies",
        "travelling",
        "health",
        "companies_brands"
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return ["soc-pokec-profiles.txt.gz", "soc-pokec-relationships.txt.gz"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        if not all([osp.exists(f) for f in self.raw_paths]):
            for url in self.urls:
                path = download_url(url, self.raw_dir)
                extract_zip(path, self.raw_dir)

    def process(self):
        dfn = pd.read_csv(self.raw_paths[0], sep = "\t", names = self.node_fields, nrows = None)
        dfe = pd.read_csv(self.raw_paths[1], sep = "\t", names = ["source", "target"], nrows = None)
        dfn = dfn.sort_index()
        for time_field in ['registration', 'last_login']:
            dfn[time_field] = pd.to_datetime(dfn[time_field], errors='coerce')
            fallback_date = pd.Timestamp('2000-01-01 00:00:00.0')
            dfn[time_field] = dfn[time_field].fillna(fallback_date)
            dfn[time_field] = dfn[time_field].astype('int64') // 10**9
        dfn['AGE'] = dfn['AGE'].replace(0, 23)
        dfn['main_area'] = dfn['region'].str.split(',').str[0]
        dfn['main_area'] = dfn['main_area'].fillna('nan')
        
        entry_dict = {}
        for column in self.edge_features:
            entries = dfn[column].unique()
            entry_set = set()
            for line in entries:
                if line == line: # Check for nan
                    hrefs = re.findall(r'href="/klub/([^"]+)"', line)
                    entry_set |= set(hrefs)
            entry_dict[column] = {entry: idx for idx, entry in enumerate(entry_set)}

        def extract_ids(entry_name, html_data):
            hrefs = re.findall(r'href="/klub/([^"]+)"', html_data)
            return [entry_dict[entry_name].get(name) for name in hrefs if name in entry_dict[entry_name]]

        edge_index_dict = {}
        for column in self.edge_features:
            print(f'Processing {column} ...')
            edge_index = []
            users_data = []
            for idx, line in enumerate(tqdm(dfn[column])):
                if line == line: # Check for nan
                    ids = extract_ids(column, line)
                    users_data.append((idx, ids))

            user_ids = [user_id for user_id, movie_ids in users_data for _ in range(len(movie_ids))]
            movie_ids = [movie_id for _, movie_ids in users_data for movie_id in movie_ids]

            # Convert lists to tensor
            user_tensor = torch.tensor(user_ids, dtype=torch.long)
            movie_tensor = torch.tensor(movie_ids, dtype=torch.long)

            # Create edge_index tensor
            edge_index = torch.stack([user_tensor, movie_tensor])
            edge_index_dict[column] = edge_index
        
        # all_text_dict = torch.load('/nobackup/users/junhong/Data/pokec/embedding.pt')
        # x = torch.zeros((len(dfn), 768))
        # for column in self.node_features:
        #     print(f'Processing {column} ...')
        #     text_dict = all_text_dict[column]
        #     for idx, text in enumerate(tqdm(dfn[column])):
        #         if text == text: # Check for nan
        #             x[idx] += text_dict[text]
        
        user_numeric_features = ['public', 'completion_percentage', 'AGE'] # 'registration', 'last_login'
        user_categorical_features = ['main_area']
        print("Getting user numerical features...")
        user_num_feats = normalize(get_numerical_features(dfn, user_numeric_features))[2]
        print("Getting user categorical features...")
        user_cat_feats = get_categorical_features(dfn, user_categorical_features)
        print("Getting user text features...")
        user_text_feats = dfn[self.node_fields[7:-1]].map(lambda x: 1 if isinstance(x, str) and x != '' else 0).values
        user_text_feats = torch.from_numpy(user_text_feats).float()
        
        data = HeteroData()

        data['user'].num_nodes = len(dfn)
        data['user'].y = torch.tensor(dfn['gender'].fillna(-1).values, dtype=torch.long)
        data['user'].x = torch.cat((user_num_feats, user_cat_feats, user_text_feats), dim=-1)
        for column, entry in entry_dict.items():
            data[column].num_nodes = len(entry_dict[column])

        source_tensor = torch.tensor(dfe['source'].values - 1, dtype=torch.long)
        target_tensor = torch.tensor(dfe['target'].values - 1, dtype=torch.long)
        data['user', 'has_friend', 'user'].edge_index = torch.stack([source_tensor, target_tensor], dim=0)
        for column, entry in entry_dict.items():
            data['user', 'lists', column].edge_index = edge_index_dict[column]

        indices = torch.where(data['user'].y != -1)[0]
        train_size = 0.5 # 50% train
        val_size = 0.5 # 25% val, 25% test
        train_idx, temp_idx = train_test_split(indices, train_size=train_size)
        val_idx, test_idx = train_test_split(temp_idx, train_size=val_size)
        train_mask = index_to_mask(train_idx, data['user'].num_nodes)
        val_mask = index_to_mask(val_idx, data['user'].num_nodes)
        test_mask = index_to_mask(test_idx, data['user'].num_nodes)
        data['user'].train_mask = train_mask
        data['user'].val_mask = val_mask
        data['user'].test_mask = test_mask
        data.validate()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_paths[0])


POKEC_URLS = {
    "profiles": "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz",
    "relations": "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
}


def download_file(url, out_path):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    if osp.exists(out_path):
        print(f"[INFO] Using existing file {out_path}")
        return

    import urllib.request
    print(f"[INFO] Downloading {url}")
    urllib.request.urlretrieve(url, out_path)


def gunzip(src, dst):
    with gzip.open(src, 'rb') as f_in:
        with open(dst, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


POKEC_PROFILE_URL = "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz"
POKEC_REL_URL     = "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"

class PokecHomoSGDataset(InMemoryDataset):
    """
    SGFormer 对齐版本的 Pokec-homo 数据集（InMemoryDataset）
    - 特征处理与 SGFormer 保持一致
    - 仅使用 SNAP 原始 60 列
    - 文本列用 boolean exist 代替（SGFormer 使用的方式）
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    @property
    def raw_file_names(self):
        return ["soc-pokec-profiles.txt.gz", "soc-pokec-relationships.txt.gz"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    # ------------------------------------------------------------------
    def download(self):
        print("[Pokec] Downloading raw files...")
        download_url(POKEC_PROFILE_URL, self.raw_dir)
        download_url(POKEC_REL_URL, self.raw_dir)

    # ------------------------------------------------------------------
    def process(self):
        print("[Pokec] Loading profiles...")
        profiles = pd.read_csv(
            self.raw_paths[0], sep='\t', header=None, names=[
                "user_id","public","completion_percentage","gender","region",
                "last_login","registration","AGE","body","I_am_working_in_field",
                "spoken_languages","hobbies","I_most_enjoy_good_food","pets",
                "body_type","my_eyesight","eye_color","hair_color","hair_type",
                "completed_level_of_education","favourite_color","relation_to_smoking",
                "relation_to_alcohol","sign_in_zodiac","on_pokec_i_am_looking_for",
                "love_is_for_me","relation_to_casual_sex","my_partner_should_be",
                "marital_status","children","relation_to_children","I_like_movies",
                "I_like_watching_movie","I_like_music","I_mostly_like_listening_to_music",
                "the_idea_of_good_evening","I_like_specialties_from_kitchen","fun",
                "I_am_going_to_concerts","my_active_sports","my_passive_sports",
                "profession","I_like_books","life_style","music","cars","politics",
                "relationships","art_culture","hobbies_interests","science_technologies",
                "computers_internet","education","sport","movies","travelling",
                "health","companies_brands","more"
            ]
        )

        print("[Pokec] Loading relationships...")
        edges = pd.read_csv(self.raw_paths[1], sep='\t', names=["src","dst"])
        edges -= 1  # zero-based index

        # ==============================================================
        # SGFormer 特征处理方式
        # ==============================================================

        # 1. 数值特征
        num_feat = profiles[["public", "completion_percentage", "AGE"]].fillna(0).to_numpy()

        # 2. 时间戳处理
        for col in ["registration", "last_login"]:
            profiles[col] = pd.to_datetime(profiles[col], errors='coerce')
            profiles[col] = profiles[col].astype("int64").fillna(0)

        time_feat = profiles[["registration", "last_login"]].to_numpy().astype(float)

        # 3. region 简化
        region_s = profiles["region"].fillna("unknown").apply(lambda x: x.split(",")[0])
        region_onehot = pd.get_dummies(region_s).to_numpy()

        # 4. 所有文本列 → 是否存在
        text_cols = profiles.columns[8:]  # body ~ more
        text_exist = profiles[text_cols].notna().astype(int).to_numpy()

        # 组合最终特征
        x = np.hstack([num_feat, time_feat, region_onehot, text_exist])
        x = torch.tensor(x, dtype=torch.float)

        # label = gender
        y = torch.tensor(profiles["gender"].fillna(-1).to_numpy(), dtype=torch.long)

        # ------------------------------------------------------------
        # edge_index (SGFormer 使用无向图)
        # ------------------------------------------------------------
        edge_index = torch.tensor(edges.values.T, dtype=torch.long)
        edge_index = to_undirected(edge_index)

        # ------------------------------------------------------------
        # Train / Val / Test split（SGFormer 标准）
        # ------------------------------------------------------------
        valid_nodes = torch.where(y != -1)[0]
        N = len(valid_nodes)
        n_train = int(N * 0.5)
        n_val = int(N * 0.25)

        perm = torch.randperm(N)
        train_idx = valid_nodes[perm[:n_train]]
        val_idx = valid_nodes[perm[n_train:n_train+n_val]]
        test_idx = valid_nodes[perm[n_train+n_val:]]

        train_mask = index_to_mask(train_idx, x.size(0))
        val_mask   = index_to_mask(val_idx, x.size(0))
        test_mask  = index_to_mask(test_idx, x.size(0))

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        torch.save(self.collate([data]), self.processed_paths[0])