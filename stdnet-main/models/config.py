"""
Backbone configurations

Author: Zhuo Su
Date: March 16, 2023
"""

from .ops import createConvFunc

backbones = {
    'baseline': {
        'layer0':  'cv',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'sdnet': {
        'layer0':  ['cv', 'cd', 'ad'],
        'layer1':  ['cv', 'cd', 'ad'],
        'layer2':  ['cv', 'cd', 'ad'],
        'layer3':  ['cv', 'cd', 'ad', 'rd'],
        'layer4':  ['cv', 'cd', 'ad'],
        'layer5':  ['cv', 'cd', 'ad'],
        'layer6':  ['cv', 'cd', 'ad'],
        'layer7':  ['cv', 'cd', 'ad', 'rd'],
        'layer8':  ['cv', 'cd', 'ad'],
        'layer9':  ['cv', 'cd', 'ad'],
        'layer10': ['cv', 'cd', 'ad'],
        'layer11': ['cv', 'cd', 'ad', 'rd'],
        'layer12': ['cv', 'cd', 'ad'],
        'layer13': ['cv', 'cd', 'ad'],
        'layer14': ['cv', 'cd', 'ad'],
        'layer15': ['cv', 'cd', 'ad', 'rd'],
        },
    'sdnet-a': {
        'layer0':  ['cv', 'cd', 'ad'],
        'layer1':  ['cv', 'cd', 'ad'],
        'layer2':  ['cv', 'cd', 'ad'],
        'layer3':  ['cv', 'cd', 'ad', 'rd'],
        'layer4':  ['cv', 'cd', 'ad'],
        'layer5':  ['cv', 'cd', 'ad'],
        'layer6':  ['cv', 'cd', 'ad'],
        'layer7':  ['cv', 'cd', 'ad'],
        'layer8':  ['cv', 'cd', 'ad'],
        'layer9':  ['cv', 'cd', 'ad'],
        },
    }


def config_model(model):
    model_options = list(backbones.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(backbones[model]))

    diffconvs = []
    for i in range(len(backbones[model])):
        layer_name = 'layer%d' % i
        op = backbones[model][layer_name]
        diffconvs.append(op)

    return diffconvs
