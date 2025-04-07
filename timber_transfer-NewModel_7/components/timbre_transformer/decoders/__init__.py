from .decoder_v21 import Decoder

current_decoder_module = Decoder.__module__
module_name = current_decoder_module.split(".")[-1]
print("Decoder module name:", module_name)
Decoder = Decoder