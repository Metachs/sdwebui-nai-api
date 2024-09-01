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
                f'{PREFIX} '+'Variety+',
                to_bool,
                xy_grid.apply_field( f'{PREFIX}_'+'variety'),
                choices= lambda: ["On","Off"]
            ))
            
            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} '+'skip_cfg_above_sigma',
                float,
                xy_grid.apply_field( f'{PREFIX}_'+'skip_cfg_above_sigma'),
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
                f'{PREFIX} '+'add_original_image',
                to_bool,
                xy_grid.apply_field( f'{PREFIX}_'+'add_original_image'),
                choices= lambda: ["On","Off"]
            ))

            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} '+'Emotion',
                str,
                xy_grid.apply_field( f'{PREFIX}_'+'emotion'),
                choices= lambda: nai_api.augment_emotions,
                format_value = lambda p,o,x: f'Emotion: {x}'
            ))


            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} '+'Defry',
                int,
                xy_grid.apply_field( f'{PREFIX}_'+'defry'),
                choices= lambda: ["0",'1','2','3','4','5'],
                format_value = lambda p,o,x: f'Defry: {x}'
            ))

            xy_grid.axis_options.append(xy_grid.AxisOption(
                f'{PREFIX} '+'Vibe <Index> <IE/RS/On/Off> <Value>',
                str,
                parse_vb_field,
                format_value = lambda p,o,x: f'Vibe: {x}'
            ))


def tryint(value, default = None):
    try:
        value = value.strip()
        return int(value)
    except Exception as e:
        pass
    try:
        return int(float(value))
    except Exception as e:
        return default


def tryfloat(value, default = None):
    try:
        value = value.strip()
        return float(value)
    except Exception as e:
        return default
        
        
def parse_vb_field(p, x, xs):        
    for s in x.split(';' if ';' in x else ','):
        s = s.strip()
        if not s: continue
        sp = s.split()
        if len(sp) < 2:
            print("Invalid Vibe Command",s )
        idx = tryint(sp[0])
        if idx is None: 
            idx = 1
            cmd = sp[0]
            vals = sp[1]
        else:
            cmd = sp[1]
            vals = sp[2] if len(sp) > 2 else None
        val = tryfloat(vals)
        cmd = cmd.lower().strip()
        field = None
        if cmd in ['on','off']:
            field = 'VibeOn'
            val = cmd == 'on'
        elif cmd in ['ie','e','ext','extract']:
            field = 'VibeExtract'
        elif cmd in ['s','rs','str','strength']:
            field = 'VibeStrength'
        elif cmd in ['m','mode']:
            field = 'VibeMode'
            val = vals
        elif cmd in ['c','color']:
            field = 'VibeColor'
            val = vals
        
        if idx>0 and field and val is not None:
            field = f'{PREFIX}_{field}_{idx}'
            # print("VB XYX", cmd, field, val)
            setattr(p, field, val)
        else: print("Invalid Vibe Command",s )
