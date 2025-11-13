import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
import spacy
import requests
import tarfile
import os
from pathlib import Path

# Simple Vocabulary class to replace torchtext vocab
class Vocabulary:
    def __init__(self, tokens, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        
        self.itos = special_tokens.copy()
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        
        for token in tokens:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        
        self.unk_index = self.stoi.get("<unk>", 0)
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)
    
    def __call__(self, tokens):
        if isinstance(tokens, str):
            return self.stoi.get(tokens, self.unk_index)
        return [self.stoi.get(token, self.unk_index) for token in tokens]
    
    def lookup_tokens(self, indices):
        return [self.itos[idx] if idx < len(self.itos) else "<unk>" for idx in indices]
    
    def set_default_index(self, index):
        self.unk_index = index


def download_multi30k(data_dir="./data"):
    """Download Multi30k dataset"""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    urls = {
        "train": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
        "valid": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    }
    
    for split, url in urls.items():
        tar_path = data_dir / f"{split}.tar.gz"
        if not tar_path.exists():
            print(f"Downloading {split} data...")
            response = requests.get(url)
            with open(tar_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Extracting {split} data...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(data_dir)
    
    return data_dir


def load_multi30k_data(split, src_lang, tgt_lang, data_dir="./data"):
    """Load Multi30k dataset from files"""
    data_dir = Path(data_dir)
    
    split_name = "train" if split == "train" else "val"
    src_file = data_dir / f"{split_name}.{src_lang}"
    tgt_file = data_dir / f"{split_name}.{tgt_lang}"
    
    if not src_file.exists() or not tgt_file.exists():
        data_dir = download_multi30k(data_dir)
        # Re-check file paths
        src_file = data_dir / f"{split_name}.{src_lang}"
        tgt_file = data_dir / f"{split_name}.{tgt_lang}"
    
    with open(src_file, 'r', encoding='utf-8') as f:
        src_data = [line.strip() for line in f]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_data = [line.strip() for line in f]
    
    return list(zip(src_data, tgt_data))


def build_vocab_from_data(data, tokenizer, min_freq=1, special_tokens=None):
    """Build vocabulary from data"""
    counter = Counter()
    for text in data:
        tokens = tokenizer(text)
        counter.update(tokens)
    
    # Filter by min_freq
    tokens = [token for token, freq in counter.items() if freq >= min_freq]
    
    return Vocabulary(tokens, special_tokens)


def load_tokenizer(lang):
    """Load spacy tokenizer for a given language"""
    spacy_models = {
        'de': 'de_core_news_sm',
        'en': 'en_core_web_sm',
        'fr': 'fr_core_news_sm',
        'es': 'es_core_news_sm',
        'it': 'it_core_news_sm',
        'pt': 'pt_core_news_sm',
        'nl': 'nl_core_news_sm',
        'el': 'el_core_news_sm',
    }
    
    if lang in spacy_models:
        try:
            nlp = spacy.load(spacy_models[lang])
            return lambda text: [token.text for token in nlp.tokenizer(text)]
        except OSError:
            print(f"Spacy model for {lang} not found. Please run: python -m spacy download {spacy_models[lang]}")
            print("Falling back to simple whitespace tokenizer...")
    
    # Fallback to simple whitespace tokenizer
    return lambda text: text.split()


# Get data, tokenizer, text transform, vocab objs, etc. Everything we
# need to start training the model
def get_data(opts):
    
    src_lang = opts.src
    tgt_lang = opts.tgt
    
    # Define special symbols
    special_symbols = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3
    }
    
    special_tokens = list(special_symbols.keys())
    
    # Load tokenizers
    print(f"Loading tokenizers for {src_lang} and {tgt_lang}...")
    src_tokenizer = load_tokenizer(src_lang)
    tgt_tokenizer = load_tokenizer(tgt_lang)
    
    # Load training data
    print("Loading training data...")
    train_data = load_multi30k_data("train", src_lang, tgt_lang, opts.data_dir if hasattr(opts, 'data_dir') else "./data")
    valid_data = load_multi30k_data("valid", src_lang, tgt_lang, opts.data_dir if hasattr(opts, 'data_dir') else "./data")
    
    # Build vocabularies
    print("Building vocabularies...")
    src_texts = [pair[0] for pair in train_data]
    tgt_texts = [pair[1] for pair in train_data]
    
    src_vocab = build_vocab_from_data(src_texts, src_tokenizer, min_freq=1, special_tokens=special_tokens)
    tgt_vocab = build_vocab_from_data(tgt_texts, tgt_tokenizer, min_freq=1, special_tokens=special_tokens)
    
    # Set default index
    src_vocab.set_default_index(special_symbols["<unk>"])
    tgt_vocab.set_default_index(special_symbols["<unk>"])
    
    # Helper function to create text transform
    def create_text_transform(tokenizer, vocab, special_symbols):
        def transform(text):
            tokens = tokenizer(text)
            token_ids = vocab(tokens)
            return torch.cat([
                torch.tensor([special_symbols["<bos>"]], dtype=torch.long),
                torch.tensor(token_ids, dtype=torch.long),
                torch.tensor([special_symbols["<eos>"]], dtype=torch.long)
            ])
        return transform
    
    src_lang_transform = create_text_transform(src_tokenizer, src_vocab, special_symbols)
    tgt_lang_transform = create_text_transform(tgt_tokenizer, tgt_vocab, special_symbols)
    
    # Create collate function
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_lang_transform(src_sample.rstrip("\n")))
            tgt_batch.append(tgt_lang_transform(tgt_sample.rstrip("\n")))
        
        src_batch = pad_sequence(src_batch, padding_value=special_symbols["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, padding_value=special_symbols["<pad>"])
        return src_batch, tgt_batch
    
    # Create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=opts.batch, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, valid_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols


def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# Create masks for input into model
def create_mask(src, tgt, pad_idx, device):
    
    # Get sequence length
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    
    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    
    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# A small test to make sure our data loads correctly
if __name__ == "__main__":
    
    class Opts:
        def __init__(self):
            self.src = "de"
            self.tgt = "en"
            self.batch = 128
            self.data_dir = "./data"
    
    opts = Opts()
    
    train_dl, valid_dl, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols = get_data(opts)
    
    print(f"{opts.src} vocab size: {len(src_vocab)}")
    print(f"{opts.tgt} vocab size: {len(tgt_vocab)}")
    print(f"Training batches: {len(train_dl)}")
    print(f"Validation batches: {len(valid_dl)}")
    
    # Test a batch
    for src, tgt in train_dl:
        print(f"Source batch shape: {src.shape}")
        print(f"Target batch shape: {tgt.shape}")
        break
