from importlib import reload
from time import sleep

import numpy as np
import polars as pl
from plotly import express as px, graph_objects as go
import sys

sys.path.append("src")

import src.cool_search as cool
import src.cool_search_models as cm
import src.utility_functions as util

util.set_plotly_template()
