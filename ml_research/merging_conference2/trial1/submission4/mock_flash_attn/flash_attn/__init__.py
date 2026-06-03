# Dummy flash_attn module
__version__ = "2.5.8"

def flash_attn_func(*args, **kwargs):
    return None

def flash_attn_kvpacked_func(*args, **kwargs):
    return None

def flash_attn_qkvpacked_func(*args, **kwargs):
    return None

def flash_attn_varlen_func(*args, **kwargs):
    return None

class flash_attn_interface:
    @staticmethod
    def flash_attn_func(*args, **kwargs):
        return None
