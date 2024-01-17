import torch
from torch import nn

def pad_tensors(input_ids, max_length):
    """Pad input_ids tensors to max_length and truncate if needed"""
    # Convert input_ids elements to PyTorch tensor if needed
    input_ids = [tensor.to('cpu') if isinstance(tensor, np.ndarray) else tensor for tensor in input_ids]
    
    # Create a packed sequence
    packed_sequence = nn.utils.rnn.pack_padded_sequence(input_ids, [len(tensor) for tensor in input_ids], batch_first=True)

    # Pad sequences
    padded_sequence, _ = nn.utils.rnn.pad_packed_sequence(packed_sequence, total_length=max_length, batch_first=True, padding_value=0)

    return padded_sequence