class ChoraleDataset(Dataset):
 def __init__(self, data_list):
 """
 data_list : list of tuples (pitch_idxs: LongTensor[T], chord_idxs:␣
 ↪LongTensor[T], length: int)
 """
 self.data = data_list
 6
def __len__(self):
 return len(self.data)
 def __getitem__(self, idx):
 pitch_idxs, chord_idxs, length = self.data[idx]
 return pitch_idxs, chord_idxs, length
 @staticmethod
 def collate_fn(batch):
 """
 batch: list of (pitch_idxs: LongTensor[T_i], chord_idxs:␣
 ↪LongTensor[T_i], len_i)
 Returns:
 padded_pitches: (B, L) LongTensor
 padded_chords : (B, L) LongTensor
 lengths
 : (B,) LongTensor
 """
 batch_size = len(batch)
 lengths = torch.LongTensor([item[2] for item in batch])
 max_len = lengths.max().item()
 padded_pitches = torch.full((batch_size, max_len),
 fill_value=pitch2idx["<PAD>"],
 dtype=torch.long)
 padded_chords = torch.full((batch_size, max_len),
 fill_value=chord2idx["<PAD>"],
 dtype=torch.long)
 for i, (p_seq, c_seq, L) in enumerate(batch):
 padded_pitches[i, :L] = p_seq
 padded_chords[i, :L] = c_seq
 return padded_pitches, padded_chords, lengths
 # Hyperparameters for DataLoader
 BATCH_SIZE = 32
 train_dataset = ChoraleDataset(train_data)
 valid_dataset = ChoraleDataset(valid_data)
 train_loader = DataLoader(
 train_dataset,
 batch_size=BATCH_SIZE,
 shuffle=True,
 collate_fn=ChoraleDataset.collate_fn
 )
 valid_loader = DataLoader(
 7
valid_dataset,
 batch_size=BATCH_SIZE,
 shuffle=False,
 collate_fn=ChoraleDataset.collate_fn
 )