import torch
import gc


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
