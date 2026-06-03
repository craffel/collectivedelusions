# Dummy bert_padding for bypassing broken global flash_attn

def index_first_axis(*args, **kwargs):
    return None

def pad_input(*args, **kwargs):
    return None, None, None, None

def unpad_input(*args, **kwargs):
    return None
