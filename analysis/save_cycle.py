import numpy as np
import pandas as pd
import os
from pathlib import Path



def save_cycle_npy(main_path, training_time, final_losses, wasserstein):
    numpy_path = os.path.join(main_path,"results")
    Path(numpy_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(numpy_path, "final_losses"), final_losses)
    np.save(os.path.join(numpy_path, "wasserstein"), wasserstein)
    np.save(os.path.join(numpy_path, "training_time"), training_time)



def save_cycle_xls(main_path, training_time, final_losses, wasserstein, str_arcs, str_bayes_arcs):
    m_df = pd.DataFrame(np.median(final_losses,axis=-1),        index = str_arcs, columns= str_bayes_arcs)
    w_df = pd.DataFrame(np.median(wasserstein,axis=-1),index = str_arcs, columns= str_bayes_arcs)
    t_df = pd.DataFrame(np.median(training_time,axis=-1), index = str_arcs, columns= str_bayes_arcs)
 
   
    with pd.ExcelWriter(os.path.join(main_path, "results.xlsx"), engine='xlsxwriter') as writer:
        m_df.to_excel(writer,sheet_name='Final Losses')   
        w_df.to_excel(writer,sheet_name='Wasserstein')
        t_df.to_excel(writer,sheet_name='Training Time')
        
        ws_mse = writer.sheets['Final Losses']
        ws_mse.write_string(0,0,'Median')



def arcs2str(architectures, bayes_arcs, all_combis):
    str_arcs = [str(i) for i in architectures]
    if all_combis:
        str_bayes_arcs = [str(i) for i in bayes_arcs]
    else:
        str_bayes_arcs = ["experiment"]
        for i, b_arc in enumerate(bayes_arcs):
            str_arcs[i] = str_arcs[i] + str(b_arc)
    return str_arcs, str_bayes_arcs
