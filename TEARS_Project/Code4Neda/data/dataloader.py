import torch
import json
from torch.utils.data import Dataset, DataLoader

class UserGenreDataset(Dataset):
    def __init__(self, json_file, num_users=None, user_start=None, user_end=None, target_pairs=None):
        with open(json_file, 'r') as f:
            self.user_genre_dict = json.load(f)

        self.user_ids = list(self.user_genre_dict.keys())

        # Filter users based on user_start and user_end
        if user_start is not None and user_end is not None:
            self.user_ids = [uid for uid in self.user_ids if int(uid) >= user_start and int(uid) <= user_end]

        # if num_users is not None:
        #     self.user_ids = self.user_ids[:num_users]  # Limit the number of users

        self.user_genres = [self.user_genre_dict[user_id] for user_id in self.user_ids]

        # Building data considering target pairs if provided
        if target_pairs:
            target_pairs_set = {tuple(pair) for pair in target_pairs}
            self.data = [
                {"user_id": user_id, "genre": genre}
                for user_id, genres in zip(self.user_ids, self.user_genres)
                for genre in genres if (user_id, genre) in target_pairs_set
            ]
        else:
            self.data = [
                {"user_id": user_id, "genre": genre}
                for user_id, genres in zip(self.user_ids, self.user_genres)
                for genre in genres
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(json_file, batch_size=32, shuffle=False, num_users=None, user_start=None, user_end=None, target_pairs=None):
    dataset = UserGenreDataset(json_file, num_users, user_start, user_end, target_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
