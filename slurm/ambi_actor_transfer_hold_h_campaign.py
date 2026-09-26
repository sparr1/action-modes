"""Frozen 575k actor transfer with one solve every H real decisions."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from slurm.ambi_actor_transfer_campaign import main

if __name__=='__main__':
    main(default_matrix=ROOT/'configs/research/ambi_actor_transfer_hold_h_575k.json',
         default_group='actor-transfer-hold-h-575k-20260926',
         default_label='575k actor transfer hold-H | H1/2/3 J1/2/4/6/8/10 | solve every H decisions')
