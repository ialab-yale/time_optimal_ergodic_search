import yaml
import io
import numpy as onp

_tor_info = {
        'pos' : onp.array([1., 1., 1.]), 
        'r1'  : 4.,
        'r2'  : 1.,
        'rot': 0.
    }

with io.open('tor.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(_tor_info, outfile, default_flow_style=False, allow_unicode=True)

with open("tor.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)

print(data_loaded)