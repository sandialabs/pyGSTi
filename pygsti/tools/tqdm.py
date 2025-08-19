try:
    from tqdm import tqdm
    def our_tqdm(iterator, message):
        return tqdm(iterator, message)
except ImportError:
    def our_tqdm(iterator, ignore):
        return iterator
