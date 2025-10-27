"""
Experiment module
Provides various classes for time series forecasting experiments

Classes:
    Exp_Basic: Basic experiment class
    Exp_Main: Main experiment class
    Exp_Online: Online learning experiment class
    Exp_XX: XX experiment class
"""

from .exp_basic import Exp_Basic
from .exp_main import Exp_Main
from .exp_online import Exp_Online
from .exp_linearRLS import Exp_LinearRLS
from .exp_foundation import Exp_Chronos
from .exp_er import Exp_ER, Exp_DERpp
from .exp_ewc import Exp_EWC