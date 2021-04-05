# -*- coding: utf-8 -*-

from .bleu import BLEU, BLEUScore
from .chrf import CHRF, CHRFScore
from .ter import TER, TERScore

METRICS = {
    'bleu': BLEU,
    'chrf': CHRF,
    'ter': TER,
}

from .clseval import ClassifierEval, MultiClassMeasure
from .clseval import AVG_TYPES, DEF_F_BETA, DEF_AVERAGE, DEF_SMOOTH_VAL
from functools import partial

# General case
#METRICS['clseval'] = partial(ClassifierEval, smooth_method='add-k', max_order=1)
# Special cases
METRICS['macrof'] = partial(ClassifierEval, average='macro', smooth_method='add-k', max_order=1)
METRICS['microf'] = partial(ClassifierEval, average='micro', smooth_method='add-k', max_order=1)