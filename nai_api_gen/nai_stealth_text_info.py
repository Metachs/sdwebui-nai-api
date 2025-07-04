from modules import script_callbacks, shared, generation_parameters_copypaste
from modules.script_callbacks import ImageSaveParams
from modules import images,errors
from PIL import Image
import PIL
import gradio as gr
import warnings
import gzip
import math
import json

from nai_api_gen.nai_read_stealth import read_stealth
from nai_api_gen.nai_api_settings import get_set_noise_schedule
from nai_api_gen.nai_api import prompt_to_a1111, prompt_to_nai,tryfloat,get_skip_cfg_above_sigma
from nai_api_gen import nai_api
    
try:
    from modules import stealth_infotext
    has_stealth_infotext = True
except:
    has_stealth_infotext = False
    
original_read_info_from_image = None
original_resize_image = None
original_flatten = None
    
def script_setup():
    global original_read_info_from_image
    global original_resize_image
    global original_flatten
    global original_read_info_from_image_stealth
    
    script_callbacks.on_before_image_saved(add_stealth_pnginfo)
    script_callbacks.on_after_component(on_after_component_change_pnginfo_image_mode)
    
    script_callbacks.on_script_unloaded(script_unload)
    
    if(original_read_info_from_image is not None): return
    
    original_read_info_from_image = images.read_info_from_image
    original_resize_image = images.resize_image
    original_flatten = images.flatten
    
    images.read_info_from_image = read_info_from_image_stealth
    images.resize_image = stealth_resize_image
    images.flatten = stealth_flatten
    

def script_unload():
    global original_read_info_from_image
    global original_resize_image
    global original_flatten
    if original_read_info_from_image is not None: images.read_info_from_image = original_read_info_from_image
    if original_resize_image is not None: images.resize_image = original_resize_image
    if original_flatten is not None: images.flatten = original_flatten
    original_read_info_from_image=None
    original_resize_image=None
    original_flatten=None

def stealth_flatten(i,c):
    if has_stealth_pnginfo(i): return i.convert("RGB")
    return original_flatten(i,c)

def on_after_component_change_pnginfo_image_mode(component, **_kwargs):
    if type(component) is gr.Image and component.elem_id == 'pnginfo_image':
        component.image_mode = 'RGBA'

def stealth_resize_image(resize_mode, im, width, height, upscaler_name=None):
    if has_stealth_pnginfo(im): im = im.convert('RGB')
    return original_resize_image(resize_mode, im, width, height, upscaler_name)
    
def add_stealth_pnginfo(params: ImageSaveParams):    
    nai_api_png_info = shared.opts.data.get("nai_api_png_info", 'NAI Only')
    if not params.filename.endswith('.png') or params.pnginfo is None:
        return
    if shared.opts.save_init_img and shared.opts.outdir_init_images and len(shared.opts.outdir_init_images) > 1 and params.filename.startswith(shared.opts.outdir_init_images): return # Do nothing if saving to init image directory
    if params.pnginfo.get("Software", None) == "NovelAI":
        if 'Comment' in params.pnginfo: 
            try:
                js_params = json.loads(params.pnginfo['Comment'])   
                if 'reference_image_multiple' in js_params: 
                    js_params.pop('reference_image_multiple')
                    pnginfo = params.pnginfo.copy() 
                    pnginfo['Comment'] = json.dumps(js_params)
                    add_data_nai(params.image, json.dumps(pnginfo)) 
                    return
            except Exception as e:
                print(e)

        add_data_nai(params.image, json.dumps(params.pnginfo))  
        return
    if 'parameters' not in params.pnginfo:
        return
    if nai_api_png_info == 'NAI Only': return
    add_data(params, 'alpha', True)
    
def process_nai_geninfo(items):
    try:
        import json
        j = json.loads(items["Comment"])
        if items.get('parameters',None) and len(items.get('parameters'))>10:
            params = items.get('parameters', None)
            if 'sampler' in j and 'NAI sampler:' not in params: 
                params+=f", NAI sampler: {j['sampler']}"
            if 'noise_schedule' in j and 'NAI noise_schedule:' not in params: 
                params+=f", NAI noise_schedule: {j['noise_schedule']}"        
            return params, items
        geninfo=items["Comment"]
        
        if 'req_type' in j:
            mode = j['req_type']
            prompt = j.get('prompt', "")
            emotion =prompt.lower() if prompt and mode == 'emotion' and prompt.lower() in nai_api.augment_emotions else None
            defry  = j.get('defry', None)
            geninfo = f'Director Tool: {mode}\n{prompt}\nNegative prompt:\nSteps: 28, NAI augment_mode: {mode}'
            if defry is not None: geninfo += f", NAI defry: {defry}"
            if emotion is not None: geninfo += f", NAI emotion: {emotion}"
            return geninfo,items
        
        if "sampler" not in j or "Description" not in items:
            print("Unrecognized NAI Metadata")
            print (items["Comment"])
            return geninfo, items
        
        from modules import sd_samplers
        sampler = sd_samplers.samplers_map.get(j["sampler"].replace('ancestral','a'), None)
        
        prompt = items["Description"]
        negs = j.get("uc", "")
        
        v4p = j.get("v4_prompt","")
        v4n = j.get("v4_negative_prompt","")        
        
        v4Weights = v4p and shared.opts.nai_api_convertImportedWeights and shared.opts.nai_api_use_numeric_emphasis
        
        if v4p:
            if v4Weights:
                prompt = nai_api.nai_v4_to_sd(prompt)
                negs = nai_api.nai_v4_to_sd(negs)
            use_coords = v4p.get('use_coords',False)
            xcoords={0.1:'A',0.3:'B',0.5:'C',0.7:'D',0.9:'E'}
            ycoords={0.1:'1',0.3:'2',0.5:'3',0.7:'4',0.9:'5'}
            ccp = v4p.get('caption', {}).get('char_captions',{})
            ccn = v4n.get('caption', {}).get('char_captions',{})
            
            if nai_api.nai_text_tag in prompt:
                prompt = prompt.replace(nai_api.nai_text_tag,'\nText:',1)
                if not ccp: prompt+='\n'

            for c in ccp:
                prompt += '\nCHAR:'
                if use_coords:
                    cs = c.get('centers')
                    if len(cs)>0:
                        x = cs[0].get('x',0.5)
                        y = cs[0].get('y',0.5)
                        if x in xcoords and y in ycoords: prompt += f'{xcoords[x]}{ycoords[y]}:'
                        else: prompt += f'{x},{y}:'
                cap = c.get('char_caption','')                
                if v4Weights: cap = nai_api.nai_v4_to_sd(cap)
                if '#' in cap:
                    if shared.opts.data.get('enable_prompt_comments', False) and shared.opts.data.get('nai_alt_action_tag', ''):
                        cap = cap.replace('#',shared.opts.data.get('nai_alt_action_tag'))
                prompt += cap
                
            if any(c.get('char_caption',None) for c in ccn):
                for c in ccn:
                    negs += '\nCHAR:'
                    cap = c.get('char_caption','')
                    if v4Weights: cap = nai_api.nai_v4_to_sd(cap)
                    negs += cap
                negs +='\n'
                
            if ccp: prompt+='\n'
        
        if not v4Weights and shared.opts.nai_api_convertImportedWeights: 
            try:
                p = prompt_to_a1111(prompt)
                n = prompt_to_a1111(negs)
                if shared.opts.data.get('nai_verbose_logging', False):
                    p2=prompt_to_nai(prompt,True)
                    n2=prompt_to_nai(negs,True)
                    if p2 != prompt: print(f"Bad conversion:\n'{prompt}'\n\n'{p2}'\n\n'{p}'")
                    if n2 != negs: print(f"Bad conversion:\n'{negs}'\n\n'{n2}'\n\n'{n}'")
                prompt = p
                negs = n
            except Exception as e:
                errors.display(e,'Converting NAI Weights')
                
        if sampler is None: sampler = 'DDIM' if 'ddim' in j["sampler"].lower() else 'Euler a'        
        geninfo = f'{prompt}\nNegative prompt: {negs}\nSteps: {j["steps"]}, Sampler: {sampler}, CFG scale: {j["scale"]}, Seed: {j["seed"]}, Size: {j["width"]}x{j["height"]}'
        
    except Exception as e:
        errors.display(e,'Reading NAI Metadata')
        print(items["Comment"])
        return geninfo,items
        
    PREFIX = "NAI "    
    def add(s="", quote =False, name = None,value = None ):
        nonlocal geninfo
        v = value or j.get(s,None)
        if v: 
            if name is None:
                geninfo+=f', {PREFIX}{s}: '  
            else:
                geninfo+=f', {name}: '  
                
            v = str(v)
            if quote: 
                if '"' in v and '\\"'not in v:
                    print (f"TEXT INFO ISSUE: unescaped quotation mark in string! \n\t{v}")
                geninfo += f'"{v}"'
            else: geninfo += f'{v}'
        return v
            
    add('enable',value = True)
    add('uncond_scale')
    add('cfg_rescale')
    add('sampler',value="Auto")
    
    noise_schedule = get_set_noise_schedule(j["sampler"], j.get("noise_schedule", ""))
    if noise_schedule: add("noise_schedule", value = noise_schedule )
    
    add('dynamic_thresholding')
    ens =add('extra_noise_seed')
    
    if ens and int(ens) != int(j["seed"]):
        add(name = 'Variation seed', value = ens)
        add(name = 'Variation seed strength', value = 1)        

    leg = add('legacy_v3_extend')
    if leg is None: add('legacy_v3_extend',value="true")
    add('add_original_image')
    add('noise')
    add('strength',"Denoising strength")
    model = items.get('Source',"")
    
    if model ==  "NovelAI Diffusion V4 F6E18726" in model: model = nai_api.NAIv4cp 
    elif model ==  "NovelAI Diffusion V4.5 B5A2A797" in model: model = nai_api.NAIv45cp 
    elif model ==  "NovelAI Diffusion V4 4F49EC75" in model: model = nai_api.NAIv4 
    elif model ==  "Stable Diffusion F1022D28": model = nai_api.NAIv2
    elif model ==  "Stable Diffusion XL 9CC2F394": model = nai_api.NAIv3f
    elif model ==  "NovelAI Diffusion V4.5 B9F340FD": model = nai_api.NAIv45
    elif model ==  "NovelAI Diffusion V4.5 1229B44F": model = nai_api.NAIv45
    else: model = nai_api.NAIv3
        
    add('model',value = model)

    skip_cfg_above_sigma = tryfloat(j.get('skip_cfg_above_sigma', None))
    if skip_cfg_above_sigma:
        epsilon = .0001
        if abs(get_skip_cfg_above_sigma(tryfloat(j["width"]), tryfloat(j["height"]),model) - skip_cfg_above_sigma) < epsilon:
            add('variety', value = True)
        else: 
            add('skip_cfg_above_sigma')        

    legv4 = add('legacy_uc')
    if 'NovelAI Diffusion V4' in model and legv4 is None: add('legacy_uc',value="true")
    
    add('smea',quote=True, value = "DYN" if str(j.get('sm_dyn','false')).lower() =='true' else "SMEA" if str(j.get('sm','false')).lower()=='true' else "Off")
    
    for i,e in enumerate(j.get('reference_image_multiple',[])):
        add(f'Vibe ID {i+1}', value = e,quote=True)
        add(f'Vibe On {i+1}', value = True,quote=True)
        
    for i,e in enumerate(j.get('reference_strength_multiple',[])):
        add(f'Vibe Strength {i+1}', value = e,quote=True)
        
    for i,e in enumerate(j.get('reference_information_extracted_multiple',[])):
        add(f'Vibe IE {i+1}', value = e,quote=True)
        
    return geninfo, items


def add_data_nai(image,text, mode='alpha', compressed=True):    
    binary_data = prepare_data(text, mode, compressed)
    if mode == 'alpha':
        image.putalpha(255)
    width, height = image.size
    pixels = image.load()
    index = 0
    end_write = False
    for x in range(width):
        for y in range(height):
            if index >= len(binary_data):
                end_write = True
                break
            values = pixels[x, y]
            if mode == 'alpha':
                r, g, b, a = values
            else:
                r, g, b = values
            if mode == 'alpha':
                a = (a & ~1) | int(binary_data[index])
                index += 1
            else:
                r = (r & ~1) | int(binary_data[index])
                if index + 1 < len(binary_data):
                    g = (g & ~1) | int(binary_data[index + 1])
                if index + 2 < len(binary_data):
                    b = (b & ~1) | int(binary_data[index + 2])
                index += 3
            pixels[x, y] = (r, g, b, a) if mode == 'alpha' else (r, g, b)
        if end_write:
            break

def prepare_data(params, mode='alpha', compressed=False):
    signature = f"stealth_{'png' if mode == 'alpha' else 'rgb'}{'info' if not compressed else 'comp'}"
    binary_signature = ''.join(format(byte, '08b') for byte in signature.encode('utf-8'))
    param = params.encode('utf-8') if not compressed else gzip.compress(bytes(params, 'utf-8'))
    binary_param = ''.join(format(byte, '08b') for byte in param)
    binary_param_len = format(len(binary_param), '032b')
    return binary_signature + binary_param_len + binary_param

def add_data(params, mode='alpha', compressed=False):
    binary_data = prepare_data(params.pnginfo['parameters'], mode, compressed)
    if mode == 'alpha':
        params.image.putalpha(255)
    width, height = params.image.size
    pixels = params.image.load()
    index = 0
    end_write = False
    for x in range(width):
        for y in range(height):
            if index >= len(binary_data):
                end_write = True
                break
            values = pixels[x, y]
            if mode == 'alpha':
                r, g, b, a = values
            else:
                r, g, b = values
            if mode == 'alpha':
                a = (a & ~1) | int(binary_data[index])
                index += 1
            else:
                r = (r & ~1) | int(binary_data[index])
                if index + 1 < len(binary_data):
                    g = (g & ~1) | int(binary_data[index + 1])
                if index + 2 < len(binary_data):
                    b = (b & ~1) | int(binary_data[index + 2])
                index += 3
            pixels[x, y] = (r, g, b, a) if mode == 'alpha' else (r, g, b)
        if end_write:
            break

def has_stealth_pnginfo(image):
    if image.mode != 'RGBA': return False
    width, height = image.size
    pixels = image.load()
    buffer_a = ''
    index_a = 0
    for x in range(width):
        for y in range(height):
            r, g, b, a = pixels[x, y]
            buffer_a += str(a & 1)
            index_a += 1
            if index_a == len('stealth_pnginfo') * 8:
                decoded_sig = bytearray(int(buffer_a[i:i + 8], 2) for i in
                                        range(0, len(buffer_a), 8)).decode('utf-8', errors='ignore')
                if decoded_sig in {'stealth_pnginfo', 'stealth_pngcomp'}:
                    return True
                else:
                    return False
    return False

def dummy_stealth_read(image):
    return None

def read_info_from_image_stealth(image):
    try:    
        if image.info is not None and image.info.get("Software", None) == "NovelAI":
            if image.info.get('parameters',None) and len(image.info['parameters']) > 10:
                # Image has both A1111 and NAI metadata, remove NAI Software entry so A1111 reads it's own metadata.
                image.info.pop("Software")
                geninfo, items = original_read_info_from_image(image)
                items["Software"] = "NovelAI"
                items['parameters'] = image.info['parameters']
                return process_nai_geninfo(items)
            return process_nai_geninfo(image.info)            
        geninfo, items = original_read_info_from_image(image)
        if geninfo is not None:
            if items is not None and items.get("Software", None) == "NovelAI": return process_nai_geninfo(items)
            try:
                items = json.loads(geninfo)
                if items is not None and items.get("Software", None) == "NovelAI": return process_nai_geninfo(items)
            except:
                pass
            return geninfo, items
                
    except Exception as e:
        errors.display(e,'read_info_from_image_stealth')
        geninfo, items = None, {}
        
    if not shared.opts.data.get("nai_api_png_info_read", True) or has_stealth_infotext: return geninfo, items
    
    try:
        geninfo = read_stealth(image)
        try:
            items = json.loads(geninfo)
        except:
            pass
        if items and items.get("Software", None) == "NovelAI": 
            return process_nai_geninfo(items)
    except Exception as e:
        errors.display(e,'read_info_from_image_stealth')

    return geninfo,items