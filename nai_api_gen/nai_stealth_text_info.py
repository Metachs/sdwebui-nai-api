# Modified Unlicensed code nabbed from an abandoned repo 
# https://github.com/ashen-sensored/sd_webui_stealth_pnginfo
# Code remains unlicensed 

from modules import script_callbacks, shared, generation_parameters_copypaste
from modules.script_callbacks import ImageSaveParams
import gradio as gr
from modules import images
from PIL import Image
from gradio import processing_utils
import PIL
import warnings
import gzip
import json

from nai_api_gen.nai_api import get_set_noise_schedule, prompt_to_a1111, prompt_to_nai
from nai_api_gen import nai_api
    
original_read_info_from_image = None
original_resize_image = None
original_flatten = None
    
def script_setup():
    global original_read_info_from_image
    global original_resize_image
    global original_flatten
    
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
    if params.pnginfo.get("Software", None) == "NovelAI":
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
        
        from modules import sd_samplers
        sampler = sd_samplers.samplers_map.get(j["sampler"].replace('ancestral','a'), None)
        
        prompt = items["Description"]
        negs = j["uc"]
        
        if shared.opts.data.get('nai_api_convertImportedWeights', True): 
            try:
                prompt = prompt_to_a1111(prompt)
                negs = prompt_to_a1111(negs)
                if shared.opts.data.get('nai_verbose_logging', False):
                    p2=prompt_to_nai(prompt,True)
                    n2=prompt_to_nai(negs,True)
                    if p2 != items["Description"]:print(f"Bad conversion:\n'{items['Description']}'\n\n'{p2}'\n\n'{prompt}'")
                    if n2 != j["uc"] :print(f"Bad conversion:\n'{j['uc']}'\n\n'{n2}'\n\n'{negs}'")                    
            except Exception as e:
                print("Error converting NAI Prompts: ",e)
                
        if sampler is None: sampler = 'DDIM' if 'ddim' in j["sampler"].lower() else 'Euler a'        
        geninfo = f'{prompt}\nNegative prompt: {negs}\nSteps: {j["steps"]}, Sampler: {sampler}, CFG scale: {j["scale"]}, Seed: {j["seed"]}, Size: {j["width"]}x{j["height"]}'
        
    except Exception as e:
        print (e)
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
    
    if model ==  "Stable Diffusion 3B3287AF": model = nai_api.NAIv1 
    elif model ==  "Stable Diffusion F4D50568": model = nai_api.NAIv1c
    elif model ==  "Stable Diffusion F64BA557": model = nai_api.NAIv1f
    elif model ==  "Stable Diffusion F1022D28": model = nai_api.NAIv2
    else: model = nai_api.NAIv3
        
    add('model',value = model)    
    
    add('smea',quote=True, value = "DYN" if str(j.get('sm_dyn','false')).lower() =='true' else "SMEA" if str(j.get('sm','false')).lower()=='true' else "Off")
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

def read_info_from_image_stealth(image,force_stealth = False):
    # Standard Metadata
    try:    
        if image.info is not None and image.info.get("Software", None) == "NovelAI":
            if 'parameters' in image.info and image.info['parameters'] and image.info['parameters']!="None":
                # Image has both A1111 and NAI metadata, remove NAI Software entry so A1111 reads it's own metadata.
                image.info.pop("Software")
                return original_read_info_from_image(image)            
            return process_nai_geninfo(image.info)            
        geninfo, items = original_read_info_from_image(image)        
        if geninfo is not None: return geninfo, items        
    except Exception as e:
        print(e)
        
    # Stealth PNG Info
    width, height = image.size
    pixels = image.load()

    has_alpha = True if image.mode == 'RGBA' else False
    mode = None
    compressed = False
    binary_data = ''
    buffer_a = ''
    buffer_rgb = ''
    index_a = 0
    index_rgb = 0
    sig_confirmed = False
    confirming_signature = True
    reading_param_len = False
    reading_param = False
    read_end = False
    for x in range(width):
        for y in range(height):
            if has_alpha:
                r, g, b, a = pixels[x, y]
                buffer_a += str(a & 1)
                index_a += 1
            else:
                r, g, b = pixels[x, y]
            buffer_rgb += str(r & 1)
            buffer_rgb += str(g & 1)
            buffer_rgb += str(b & 1)
            index_rgb += 3
            if confirming_signature:
                if index_a == len('stealth_pnginfo') * 8:
                    decoded_sig = bytearray(int(buffer_a[i:i + 8], 2) for i in
                                            range(0, len(buffer_a), 8)).decode('utf-8', errors='ignore')
                    if decoded_sig in {'stealth_pnginfo', 'stealth_pngcomp'}:
                        confirming_signature = False
                        sig_confirmed = True
                        reading_param_len = True
                        mode = 'alpha'
                        if decoded_sig == 'stealth_pngcomp':
                            compressed = True
                        buffer_a = ''
                        index_a = 0
                    else:
                        read_end = True
                        break
                elif index_rgb == len('stealth_pnginfo') * 8:
                    decoded_sig = bytearray(int(buffer_rgb[i:i + 8], 2) for i in
                                            range(0, len(buffer_rgb), 8)).decode('utf-8', errors='ignore')
                    if decoded_sig in {'stealth_rgbinfo', 'stealth_rgbcomp'}:
                        confirming_signature = False
                        sig_confirmed = True
                        reading_param_len = True
                        mode = 'rgb'
                        if decoded_sig == 'stealth_rgbcomp':
                            compressed = True
                        buffer_rgb = ''
                        index_rgb = 0
            elif reading_param_len:
                if mode == 'alpha':
                    if index_a == 32:
                        param_len = int(buffer_a, 2)
                        reading_param_len = False
                        reading_param = True
                        buffer_a = ''
                        index_a = 0
                else:
                    if index_rgb == 33:
                        pop = buffer_rgb[-1]
                        buffer_rgb = buffer_rgb[:-1]
                        param_len = int(buffer_rgb, 2)
                        reading_param_len = False
                        reading_param = True
                        buffer_rgb = pop
                        index_rgb = 1
            elif reading_param:
                if mode == 'alpha':
                    if index_a == param_len:
                        binary_data = buffer_a
                        read_end = True
                        break
                else:
                    if index_rgb >= param_len:
                        diff = param_len - index_rgb
                        if diff < 0:
                            buffer_rgb = buffer_rgb[:diff]
                        binary_data = buffer_rgb
                        read_end = True
                        break
            else:
                # impossible
                read_end = True
                break
        if read_end:
            break
    if sig_confirmed and binary_data != '':
        # Convert binary string to UTF-8 encoded text
        byte_data = bytearray(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8))
        try:
            if compressed:
                decoded_data = gzip.decompress(bytes(byte_data)).decode('utf-8')                
            else:
                decoded_data = byte_data.decode('utf-8', errors='ignore')
            geninfo = decoded_data
            try:
                import json
                items = json.loads(decoded_data)
                if items is not None and items.get("Software", None) == "NovelAI": 
                    return process_nai_geninfo(items)                
            except Exception as e:
                print (e)
        except Exception as e:
            print (e)
            
    return geninfo, items
