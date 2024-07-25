from modules import scripts, script_callbacks, shared
import gradio as gr
import os
from nai_api_gen import nai_api
        
def to_bool(v: str):
    try:
        if len(v) == 0: return False
        v = v.lower()
        if 'true' in v: return True
        if 'false' in v: return False
        if 'on' in v: return True
        if 'off' in v: return False
        if 'yes' in v: return True
        if 'no' in v: return False
        w = int(v)
        return bool(w)
    except:
        raise Exception(f"Invalid bool")

PREFIX = "NAI"


def script_setup():    
    script_callbacks.on_before_ui(xyz_setup)

def xyz_setup():
    for script in scripts.scripts_data:
        if os.path.basename(script.path) == 'xyz_grid.py':
            xy_grid = script.module
            if hasattr(xy_grid,"_NAI_GRID_OPTIONS"): return
            xy_grid._NAI_GRID_OPTIONS=True
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} API',
                to_bool,
                xy_grid.apply_field(f'{PREFIX}_enable'),
                choices= lambda: ["On","Off"]
            ))
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} SMEA/DYN',
                str,
                xy_grid.apply_field(f'{PREFIX}_smea'),
                choices= lambda: ["Off","SMEA","DYN"]
            ))
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} Model',
                str,
                xy_grid.apply_field( f'{PREFIX}_model'),
                choices= lambda: nai_api.nai_models
            ))
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} Sampler',
                str,
                xy_grid.apply_field( f'{PREFIX}_sampler'),
                choices= lambda: nai_api.NAI_SAMPLERS
            ))
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} Decrisper (dynamic_thresholding)',
                to_bool,
                xy_grid.apply_field( f'{PREFIX}_'+'dynamic_thresholding'),
                choices= lambda: ["On","Off"]
            ))
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} '+'CFG Rescale ',
                float,
                xy_grid.apply_field( f'{PREFIX}_'+'cfg_rescale'),
            ))
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} Noise Schedule',
                str,
                xy_grid.apply_field( f'{PREFIX}_'+'noise_schedule'),
                choices= lambda: ["recommended","exponential","polyexponential","karras","native"]
            ))

            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} '+'extra_noise ',
                float,
                xy_grid.apply_field( f'{PREFIX}_'+'extra_noise'),
            ))
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} '+'add_original_image ',
                to_bool,
                xy_grid.apply_field( f'{PREFIX}_'+'add_original_image'),
                choices= lambda: ["On","Off"]
            ))
