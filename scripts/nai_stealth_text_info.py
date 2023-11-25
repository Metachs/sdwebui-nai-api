# Modified Unlicensed code nabbed from an abandoned repo 
# https://github.com/ashen-sensored/sd_webui_stealth_pnginfo

from modules import script_callbacks, shared, generation_parameters_copypaste
from modules.script_callbacks import ImageSaveParams
import gradio as gr
from modules import images
from PIL import Image
from gradio import processing_utils
import PIL
import warnings
import gzip

#Fix for not saving PNGInfo on saved images, probably a better way but this was the easiest

def add_stealth_pnginfo(params: ImageSaveParams):
    
    nai_api_png_info = shared.opts.data.get("nai_api_png_info", 'NAI Only')
    # stealth_pnginfo_mode = shared.opts.data.get('stealth_pnginfo_mode', 'alpha')
    # stealth_pnginfo_compressed = shared.opts.data.get("stealth_pnginfo_compression", True)
    
    if nai_api_png_info == 'Disable':
        return
    if not params.filename.endswith('.png') or params.pnginfo is None:
        return
    if 'parameters' not in params.pnginfo:
        return
    geninfo, items = original_read_info_from_image(params.image)
    if items.get("Software", None) == "NovelAI":
        import json
        add_data_nai(params.image, json.dumps(items))
        #params.pnginfo = items
        def move(s):
            nonlocal params
            params.pnginfo[s] = items[s]
        move('Title')
        move('Description')
        move('Software')
        move('Source')
        move('Generation time')
        move('Comment')
        return
    if nai_api_png_info == 'NAI Only': return
    add_data(params, stealth_pnginfo_mode, stealth_pnginfo_compressed)


def process_nai_geninfo(geninfo, items):
    if items is None or items.get("Software", None) != "NovelAI":
        return geninfo,items
    try:
        import json
        j = json.loads(items["Comment"])
        
        from modules import sd_samplers
        
        sampler = sd_samplers.samplers_map.get(j["sampler"], None)
        if sampler is None:  'DDIM' if 'ddim' in sampler.lower() else 'Euler a'        
        geninfo = f'{items["Description"]}\nNegative prompt: {j["uc"]}\nSteps: {j["steps"]}, Sampler: {sampler}, CFG scale: {j["scale"]}, Seed: {j["seed"]}, Size: {j["width"]}x{j["height"]}'
    except Exception as e:
        print (e)
        return geninfo,items
    PREFIX = "NAI "    
    def add(s, quote =False, name = None,value = None ):
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
    
    from scripts.nai_api import get_set_noise_schedule
    noise_schedule = get_set_noise_schedule(j["sampler"], j.get("noise_schedule", ""))
    if noise_schedule: add("noise_schedule", value = noise_schedule )
    
    add('dynamic_thresholding')
    add('extra_noise_seed')
    add('add_original_image')
    add('noise')
    add('strength',"Denoising strength")
    from scripts import nai_api
    model = items.get('Source',"")
    
    if model ==  "Stable Diffusion 3B3287AF": model = nai_api.NAIv1 
    elif model ==  "Stable Diffusion F4D50568": model = nai_api.NAIv1c
    elif model ==  "Stable Diffusion F64BA557": model = nai_api.NAIv1f
    elif model ==  "Stable Diffusion F1022D28": model = nai_api.NAIv2
    else: model = nai_api.NAIv3
        
    add('model',value = model)
    
    s = add('sm')
    sd = add('sm_dyn')
    
    def is_true(v):
        if type(s) == str: return s.lower() == "true" 
        return v
        
    add('smea',quote=True, value = "DYN" if is_true(sd) else "SMEA" if is_true(s) else "Off")
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



def read_info_from_image_stealth(image):
    geninfo, items = original_read_info_from_image(image)
    # possible_sigs = {'stealth_pnginfo', 'stealth_pngcomp', 'stealth_rgbinfo', 'stealth_rgbcomp'}

    # respecting original pnginfo
    if geninfo is not None:
        return process_nai_geninfo(geninfo, items)

    # trying to read stealth pnginfo
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
                return process_nai_geninfo(geninfo, items)
                
            except Exception as e:
                print (e)
        except Exception as e:
            print (e)
            
    return geninfo, items

def send_rgb_image_and_dimension(x):
    if isinstance(x, Image.Image):
        img = x
        if img.mode == 'RGBA':
            img = img.convert('RGB')
    else:
        img = generation_parameters_copypaste.image_from_url_text(x)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

    if shared.opts.send_size and isinstance(img, Image.Image):
        w = img.width
        h = img.height
    else:
        w = gr.update()
        h = gr.update()

    return img, w, h


def on_ui_settings():
    section = ('stealth_pnginfo', "Stealth PNGinfo")
    shared.opts.add_option("stealth_pnginfo", shared.OptionInfo(
        True, "Save Stealth PNGinfo", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("stealth_pnginfo_mode", shared.OptionInfo(
        "alpha", "Stealth PNGinfo mode", gr.Dropdown, {"choices": ["alpha", "rgb"], "interactive": True},
        section=section))
    shared.opts.add_option("stealth_pnginfo_compression", shared.OptionInfo(
        True, "Stealth PNGinfo compression", gr.Checkbox, {"interactive": True}, section=section))


def custom_image_preprocess(self, x):
    if x is None:
        return x

    mask = ""
    if self.tool == "sketch" and self.source in ["upload", "webcam"]:
        assert isinstance(x, dict)
        x, mask = x["image"], x["mask"]

    assert isinstance(x, str)
    im = processing_utils.decode_base64_to_image(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = im.convert(self.image_mode)
    if self.shape is not None:
        im = processing_utils.resize_and_crop(im, self.shape)
    if self.invert_colors:
        im = PIL.ImageOps.invert(im)
    if (
            self.source == "webcam"
            and self.mirror_webcam is True
            and self.tool != "color-sketch"
    ):
        im = PIL.ImageOps.mirror(im)

    if self.tool == "sketch" and self.source in ["upload", "webcam"]:
        mask_im = None
        if mask is not None:
            mask_im = processing_utils.decode_base64_to_image(mask)

        return {
            "image": self._format_image(im),
            "mask": self._format_image(mask_im),
        }

    return self._format_image(im)


def on_after_component_change_pnginfo_image_mode(component, **_kwargs):
    if type(component) is gr.State:
        return
    if type(component) is gr.Image and component.elem_id == 'pnginfo_image':
        component.image_mode = 'RGBA'

    def clear_alpha(param):
        output_image = param['image'].convert('RGB')
        return output_image

    if type(component) is gr.Image and component.elem_id == 'img2maskimg':
        component.upload(clear_alpha, component, component)
        component.preprocess = custom_image_preprocess.__get__(component, gr.Image)


def stealth_resize_image(resize_mode, im, width, height, upscaler_name=None):
    if im.mode == 'RGBA': im = im.convert('RGB')
    return original_resize_image(resize_mode, im, width, height, upscaler_name)


original_read_info_from_image = images.read_info_from_image
images.read_info_from_image = read_info_from_image_stealth
generation_parameters_copypaste.send_image_and_dimensions = send_rgb_image_and_dimension
original_resize_image = images.resize_image
images.resize_image = stealth_resize_image

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(add_stealth_pnginfo)
script_callbacks.on_after_component(on_after_component_change_pnginfo_image_mode)