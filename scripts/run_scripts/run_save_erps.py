import numpy as np
import os
import sys
sys.path.append("../")

from tfsenc_main import process_subjects, return_stitch_index
from tfsenc_read_datum import read_datum
from tfserp_main import load_and_erp

class SandboxArgs:
  def __init__(self):
    pass

def save_erps(sid: int, electrodes,
              lags=np.arange(-10000, 10100, 200), save_dir="/scratch/gpfs/cc27/results/tfs/{sid}/erps"): 
    args = SandboxArgs()

    args.emb_df_path = f"/scratch/gpfs/cc27/results/tfs/{sid}/pickles/embeddings/gpt2-xl/full/cnxt_0008/layer_48.pkl"
    args.base_df_path = f"/scratch/gpfs/cc27/results/tfs/{sid}/pickles/embeddings/gpt2-xl/full/base_df.pkl"
    args.PICKLE_DIR = f"/scratch/gpfs/cc27/results/tfs/{sid}/pickles"
    args.pkl_identifier = "full"
    args.stitch_file = "_".join([str(sid), args.pkl_identifier, "stitch_index.pkl"])
    args.bad_convos = []
    args.normalize = "l2"
    args.emb_type = "gpt2-xl"
    args.datum_mod = "notrim-all"
    args.min_word_freq = 0
    args.exclude_nonwords = False
    args.align_with = ["gpt2-xl"]
    args.conversation_id = None
    args.electrode_file = "_".join([str(sid), "electrode_names.pkl"])
    args.sig_elec_file = None
    args.electrodes = electrodes
    args.sid = sid
    args.project_id = "tfs"
    args.conversation_id = 0
    args.lags = lags

    stitch_index = return_stitch_index(args)
    datum = read_datum(args, stitch_index)
    datum = datum.drop("embeddings", axis=1)
    electrode_info = process_subjects(args)

    erps_dict_comp = {"lags": lags}
    erps_dict_prod = {"lags": lags}
    for electrode_idx, electrode in enumerate(electrode_info.items()):
        (sid, elec_id), elec_name = electrode
        print(elec_name, electrode, f"{electrode_idx}/{len(electrode_info)}")
        try:
            erp_comp, erp_prod = load_and_erp((elec_id, elec_name), args, datum, stitch_index)
            erps_dict_comp[elec_name] = erp_comp
            erps_dict_prod[elec_name] = erp_prod
        except Exception as e:
            print(f"FAILED on {electrode}, {e}")
    
    save_dir = save_dir.format(sid=sid)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez(f"{save_dir}/erps_comp.npz", **erps_dict_comp)
    np.savez(f"{save_dir}/erps_prod.npz", **erps_dict_prod)

if __name__ == "__main__":
    for (sid, electrodes) in [(625, np.arange(1, 105)),
                              (676, np.arange(1, 125)),
                              (7170, np.arange(1, 256)),
                              (798, np.arange(1, 198))]:
        print(f"Saving ERPs for {sid}")
        save_erps(sid, electrodes)
