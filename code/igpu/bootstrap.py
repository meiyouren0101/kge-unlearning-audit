from pathlib import Path
import sys
ENGINE=Path(__file__).resolve().parent/"engine"
if str(ENGINE) not in sys.path:sys.path.insert(0,str(ENGINE))
