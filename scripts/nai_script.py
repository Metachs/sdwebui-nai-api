from modules import scripts, script_callbacks, shared, sd_samplers
import gradio as gr

import os
from PIL import Image 
import hashlib

import math
import re

import time

from scripts.nai_api import POST, LOAD, NAIGenParams, convert,prompt_to_nai,subscription_status,prompt_has_weight

import modules.processing

import modules 


import modules.images as images
from modules.processing import process_images
from modules.processing import Processed, StableDiffusionProcessingTxt2Img,StableDiffusionProcessingImg2Img,create_infotext


NAIv1 = "nai-diffusion"
NAIv1c = "safe-diffusion"
NAIv1f = "nai-diffusion-furry"
NAIv2 = "nai-diffusion-2"
NAIv3 = "nai-diffusion-3"

hashdic = {}    

NAI_SAMPLERS = ["k_euler","k_euler_ancestral","k_dpmpp_2s_ancestral","k_dpmpp_2m","ddim_v3","k_dpmpp_sde"]

def get_api_key():
    return shared.opts.data.get('nai_api_key', None)



def on_ui_settings():
    section = ('nai_api', "NAI API Generator")
    def addopt(n,o):
        if not n in shared.opts.data_labels:
            shared.opts.add_option(n,o)           
    text ="To Image "
    #modes = {"choices": [ 'Script' , 'Process' , 'Patch']}
    
    addopt('nai_api_key', shared.OptionInfo('', "NAI API Key - See https://docs.sillytavern.app/usage/api-connections/novelai/ ", gr.Textbox, section=section))
    
    addopt('nai_api_skip_checks', shared.OptionInfo(False, "Skip NAI account/subscription/Anlas checks.",gr.Checkbox, section=section))
    
    addopt('nai_api_png_info', shared.OptionInfo( 'NAI Only', "Stealth PNG Info - Read/Write Stealth PNG info for NAI images only (required to emulate NAI), All Images, or None",gr.Radio, {"choices": ['NAI Only', 'All Images', 'Disable'] }, section=section))
    
    addopt('nai_api_default_sampler', shared.OptionInfo('k_euler', "Fallback Sampler: Used when the sampler doesn't match any sampler available in NAI .",gr.Radio, {"choices": NAI_SAMPLERS }, section=section))
        
    #addopt('NAI_gen_mode', shared.OptionInfo('Script', "Mode (REQUIRES RESTART) - Script uses the Scripts dropdown. Patch does not, allowing use of other Scripts (eg XYZ Grid), but may cause conflicts with other extensions or other issues.",gr.Radio, modes, section=section))
script_callbacks.on_ui_settings(on_ui_settings)

class NAIGENScript(scripts.Script):

    def __init__(self):    
        super().__init__()
        self.NAISCRIPTNAME = "NAI"    
        self.images = []
        self.hashes = []
        self.texts = []
        self.sampler_name = None
        self.api_connected = False
        
    def title(self):
        return self.NAISCRIPTNAME


    def show(self, is_img2img):
        return False
        
    def process_inner(self, p, **kwargs):
        pass
        
    def skip_checks(self):
        return shared.opts.data.get('nai_api_skip_checks', False)
                
    def get_api_key(self):
        return shared.opts.data.get('nai_api_key', None)
    
    def connect_api(self):
        s,m = self.subscription_status_message()
        self.api_connected=s
        return s, f'{m}'

    def check_api_key(self,skip_sub=False):    
        key = self.get_api_key()
        if key is None or len(key) < 6: return False
        if skip_sub or self.skip_checks(): return True
        status, opus, points, max = subscription_status(key)
        return opus or points > 1   

    def subscription_status_message(self):
        key = self.get_api_key()
        status, opus, points, max = subscription_status(key)
        #print(f'{status} {opus} {points} {max}')
        if status == -1: return False,"[API ERROR] Missing API Key, enter in options menu"
        elif status == 401: return False,"Invalid API Key"
        elif status != 200: return False,f"[API ERROR] Error Code: {status}"
        elif not opus and points <=1:
            return True, f'[API ERROR] Insufficient points! {points}'
        return True, f'[API OK] Anlas:{points} {"Opus" if opus else ""}'
    
    def setup_sampler_name(self,p, nai_sampler):
        if nai_sampler not in NAI_SAMPLERS:
            nai_sampler = self.get_nai_sampler(p.sampler_name)
            p.sampler_name = sd_samplers.all_samplers_map.get(nai_sampler,None) or p.sampler_name
        self.sampler_name = nai_sampler
        
    
    def get_nai_sampler(self,sampler_name):
        sampler = sd_samplers.all_samplers_map.get(sampler_name)
        if sampler.name in ["DDIM","PLMS","UniPC"]: return "ddim_v3"
        for n in NAI_SAMPLERS:
            if n in sampler.aliases:
                return n
        return shared.opts.data.get('NAI_gen_default_sampler', 'k_euler')
        
    def limit_costs(self, p, nai_batch = False):
        MAXSIZE = 1048576
        if p.width * p.height > MAXSIZE:
            scale = p.width/p.height
            p.width= int(math.sqrt(MAXSIZE/scale))
            p.height= int(p.width * scale)
            print(f"Cost Limiter: Reduce dimensions to {p.width} x {p.height}")
        if nai_batch and p.batch_size > 1:
            p.n_iter *= p.batch_size
            p.batch_size = 1
            print(f" Cost Limiter: Disable Batching")
        if p.steps >28: 
            p.steps = 28
            print(f"Cost Limiter: Reduce steps to {p.steps}")

    def adjust_resolution(self, p):
        if p.width % 64 == 0 and p.height % 64 == 0: return
        print(f'Adjusting resolution from {p.width} x {p.height} to {int(p.width/64)*64} {int(p.height/64)*64}- NAI dimensions must be divisible by 64')
        p.width = int(p.width/64)*64
        p.height = int(p.height/64)*64
    
    def convert_to_nai(self, prompt, neg,convert_prompts="Always"):
        if convert_prompts != "Never":
            if convert_prompts == "Always" or prompt_has_weight(prompt): prompt = prompt_to_nai(prompt)
            if convert_prompts == "Always" or prompt_has_weight(prompt): neg = prompt_to_nai(neg)
        return prompt, neg

    def get_batch_images(self, p, getparams, save_images = False , save_suffix = "", dohash = False, query_batch_size = 0):            
        key = shared.opts.data.get('NAI_gen_key', None)
        
        nai_format = shared.opts.data.get('NAI_gen_text_info', 'NAI') == 'NAI'
        
        def infotext(i):
            iteration = int(i / p.n_iter)
            batch = i % p.batch_size            
            return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, iteration, batch)

        while len(self.images) < p.iteration*p.batch_size + p.batch_size:
            results=[]
            resultsidx=[]
            for i in range( len(self.images) , min(len(self.images) + query_batch_size,  p.n_iter*p.batch_size) ):        
                parameters = getparams(i)
                if dohash and len(parameters) < 10000 and parameters in hashdic:
                    hash = hashdic[parameters]
                    imgp = os.path.join(shared.opts.outdir_init_images, f"{hash}.png")
                    if os.path.exists(imgp):
                        print("Loading Previously Generated Image")
                        self.images.append(Image.open(imgp))
                        self.hashes.append(None)
                        self.texts.append(infotext(i))
                else:
                    self.images.append(None)
                    self.hashes.append(None)
                    results.append(POST(key, parameters, g = query_batch_size > 1))
                    resultsidx.append(i)               
                    self.texts.append("")
                
                    
            if query_batch_size > 1: 
                import grequests    
                results = grequests.map(results)
            

            for ri in range(len(results)):
                result = results[ri]
                i = resultsidx[ri]
                image,code =  LOAD(result, parameters)
                #TODO: Handle time out errors
                if image is None: continue
                if dohash:
                    hash = hashlib.md5(image.tobytes()).hexdigest()
                    self.hashes[i] = hash
                    if not getattr(p, "use_txt_init_img",False): p.extra_generation_params["txt_init_img_hash"] = hash
                    if len(parameters) < 10000: hashdic[parameters] = hash
                    if not os.path.exists(os.path.join(shared.opts.outdir_init_images, f"{hash}.png")):
                        images.save_image(image, path=shared.opts.outdir_init_images, basename=None, extension='png', forced_filename=hash, save_to_dirs=False)
                self.images[i] = image
                self.texts[i] = infotext(i)
                    

                images.save_image(image, p.outpath_samples, "", p.all_seeds[i], p.all_prompts[i], shared.opts.samples_format, info=self.texts[i], suffix=save_suffix)
                
        for i in range( p.iteration*p.batch_size, p.iteration*p.batch_size+p.batch_size):
            parameters = getparams(i)
            if self.images[i] is None:
                print("Image Failed to Load, Giving Up")
                if dohash and p.batch_size * p.n_iter == 1:  p.enable_hr = False
                self.images[i] = Image.new("RGBA",(p.width, p.height), color = "black")
        
MODE= shared.opts.data.get('NAI_gen_mode', 'Script')
#print(MODE)

class NAIGenException(Exception):
    pass
    
def process_images_patched(p):
    def FindScript(scripts):
        if scripts is None:
            print("No Scripts!")
            return None
        for script in scripts.alwayson_scripts:  
            # isinstance(script, NAIGENScript) stopped worked seemingly at random, possibly after splitting files?
            # Yes apparently each file just re-implements everything it imports, including base classes, like the fucking worst type of c++ macro bullshit and therefore is not an instance of the base class implmented in other files. Whether this is just how Python works, or if it is a quirk of the way extensions are implemented in sdwebui, it is fucking ridiculous.
            # Probably need to look up how python imports work, clearly it is not anything sane, rational or predictable, like everything else in this trainwreck of a language
            if hasattr(script, "NAISCRIPTNAME"): return script        

    if MODE != 'Patch': return modules.processing.process_images_pre_patch_4_nai(p)
    
    script = None if MODE != 'Patch' else FindScript(p.scripts)
    
    if script is None: 
        print("NAIGENScript is None")        
        return modules.processing.process_images_pre_patch_4_nai(p)
        
    p.nai_processed=None
    p.scripts.originalprocess = p.scripts.process
    if hasattr(p, "per_script_args"):
        args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
    else: 
        args = p.script_args[script.args_from:script.args_to]
    
    try:
        script.patched_process(p, *args)
        
        def process_patched(self,p, **kwargs):
            p.scripts.originalprocess(p, **kwargs)
            try:
                script.process_inner(p, *args) # Sets p.nai_processed if images have been processed
            except Exception as e:
                import modules.errors as errors
                errors.display(e, 'executing callback: {script} {process_batch}')            
            if hasattr(p,"nai_processed") and p.nai_processed is not None:
                # Make sure post process is called, at least one extension uses it to clean up from things done in process. 
                # This may also cause problems since batch process isn't called, but this seems less likely.
                p.scripts.postprocess(p, p.nai_processed) 
                raise NAIGenException

        p.scripts.process= process_patched.__get__(p.scripts, scripts.ScriptRunner)
        
        return modules.processing.process_images_pre_patch_4_nai(p)
    except NAIGenException:
        return p.nai_processed
    finally: 
        p.scripts.process=p.scripts.originalprocess
        

if MODE == 'Patch':
    if not hasattr(modules.processing, 'process_images_pre_patch_4_nai'):
        modules.processing.process_images_pre_patch_4_nai = modules.processing.process_images_inner     
        print("Patching Image Processing for NAI Generator Script, this may could potentially cause compatibility issues.")
    modules.processing.process_images_inner = process_images_patched


# def on_script_unloaded():
    # if hasattr(modules.processing, 'process_images_pre_patch_4_nai'):
        # modules.processing.process_images = modules.processing.process_images_pre_patch_4_nai 

# script_callbacks.on_script_unloaded(on_script_unloaded)

