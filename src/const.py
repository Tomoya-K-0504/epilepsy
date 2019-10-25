from pathlib import Path


ALL_LABELS = """NULL, SPSW, GPED, PLED, EYBL, ARTF, BCKG, SEIZ, FNSZ, GNSZ, SPSZ, CPSZ, ABSZ, TNSZ, CNSZ, TCSZ, 
ATSZ, MYSZ, NESZ, INTR, SLOW, EYEM, CHEW, SHIV, MUSC, ELPP, ELST""".lower().split(', ')


BONN_LABELS = {'set_A': 'none', 'set_B': 'none', 'set_C': 'interictal', 'set_D': 'interictal', 'set_E': 'ictal'}