import inspect
from diffusers import FluxPriorReduxPipeline
print(inspect.getsource(FluxPriorReduxPipeline.__call__))
