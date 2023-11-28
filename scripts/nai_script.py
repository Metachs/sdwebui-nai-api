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
        self.disabled=False
        self.failed=False
        self.failure=""
        self.running=True
        self.do_nai_post=False
        
        
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
        
    def initialize(self):
        self.failed= False
        self.failure =""
        self.disabled= False
        self.images=[]
        self.hashes=[]
        self.texts=[]
        self.nai_processed=None
        
    def comment(self, p , c):
        print (c)
        if p is None or not hasattr(p, "comment"):return        
        p.comment(c)    
        
    def fail(self, p, c):
        self.comment(p,c)
        self.failed=True
        self.failure = c
        self.disabled=True
    
    def limit_costs(self, p, nai_batch = False):
        MAXSIZE = 1048576
        if p.width * p.height > MAXSIZE:
            scale = p.width/p.height
            p.height= int(math.sqrt(MAXSIZE/scale))
            p.width= int(p.height * scale)
                
            p.width = int(p.width/64)*64
            p.height = int(p.height/64)*64
        
            self.comment(p,f"Cost Limiter: Reduce dimensions to {p.width} x {p.height}")
            
        if nai_batch and p.batch_size > 1:
            p.n_iter *= p.batch_size
            p.batch_size = 1
            self.comment(p,f" Cost Limiter: Disable Batching")
        if p.steps >28: 
            p.steps = 28
            self.comment(p,f"Cost Limiter: Reduce steps to {p.steps}")
    
    def adjust_resolution(self, p):
        #if p.width % 64 == 0 and p.height % 64 == 0 or p.width *p.height > 1792*1728: return
        width = p.width
        height = p.height        
        
        width = int(p.width/64)*64
        height = int(p.height/64)*64
        
        MAXSIZE = 1792*1728
        
        if width *height > MAXSIZE:            
            scale = width/height
            width= int(math.sqrt(MAXSIZE/scale))
            height= int(width * scale)
            width = int(p.width/64)*64
            height = int(p.height/64)*64
        
        if width == p.width and height == p.height: return
        
        self.comment(p,f'Adjusted resolution from {p.width} x {p.height} to {width} x {height}- NAI dimensions must be multiples of 64 and <= 1792x1728')
        
        p.width = width
        p.height = height
        
        
        

    
    def convert_to_nai(self, prompt, neg,convert_prompts="Always"):
        if convert_prompts != "Never":
            if convert_prompts == "Always" or prompt_has_weight(prompt): prompt = prompt_to_nai(prompt)
            if convert_prompts == "Always" or prompt_has_weight(prompt): neg = prompt_to_nai(neg)
            prompt=prompt.replace('\\(','(').replace('\\)',')')
            neg=neg.replace('\\(','(').replace('\\)',')')
        return prompt, neg

    def get_batch_images(self, p, getparams, save_images = False , save_suffix = "", dohash = False, query_batch_size = 1):          
    
        key = get_api_key()
        
        nai_format = shared.opts.data.get('NAI_gen_text_info', 'NAI') == 'NAI'
        
        def infotext(i):
            iteration = int(i / (p.n_iter*p.batch_size))
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
                elif len(p.all_prompts[i]) < 1:
                    self.images.append(None)
                    self.hashes.append(None)
                    self.texts.append("")
                    self.comment(p,f'NAI Does Not Support Empty Prompts!')
                else:
                    self.images.append(None)
                    self.hashes.append(None)
                    self.texts.append("")
                    if dohash:
                        strip = re.sub("\"image\":\".*?\"","\"image\":\"\"" ,re.sub("\"mask\":\".*?\"","\"mask\":\"\"" ,parameters))
                        self.comment(p,f'{strip}')
                    results.append(POST(key, parameters, g = query_batch_size > 1))
                    resultsidx.append(i)               
                
                    
            if query_batch_size > 1: 
                import grequests    
                results = grequests.map(results)
            

            for ri in range(len(results)):
                result = results[ri]
                i = resultsidx[ri]
                image,code =  LOAD(result, parameters)
                #TODO: Handle time out errors
                if image is None: 
                    self.texts[i] = code
                    continue
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
                
        for i in range(p.iteration*p.batch_size, p.iteration*p.batch_size+p.batch_size):
            if self.images[i] is None:
                if p.n_iter*p.batch_size == 1:
                    self.fail(p,f'Failed to retrieve image - Error Code: {self.texts[i]}')
                else: self.comment(p,f'Failed to retrieve image {i} - Error Code: {self.texts[i]}')
                print("Image Failed to Load, Giving Up")
                if dohash and p.batch_size * p.n_iter == 1:  p.enable_hr = False
                self.images[i] = Image.new("RGBA",(p.width, p.height), color = "black")
            else:
                if i == 0 and save_images:
                    import modules.paths as paths
                    with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                        processed = Processed(p, [])
                        file.write(processed.infotext(p, 0))

        
class NAIGenException(Exception):
    pass

POSTPROCESS = 0
running_scripts=[] # This script stack is probably unnecessary
    
def process_images_patched(p):
    def FindScript(scripts):
        if scripts is None:
            return None
        for script in scripts.alwayson_scripts:  
            # isinstance(script, NAIGENScript) stopped worked seemingly at random, possibly after splitting files?
            # Yes apparently each file just re-implements everything it imports, including base classes, like the fucking worst type of c++ macro bullshit and therefore is not an instance of the base class implmented in other files. Whether this is just how Python works, or if it is a quirk of the way extensions are implemented in sdwebui, it is fucking ridiculous.
            # Probably need to look up how python imports work, clearly it is not anything sane, rational or predictable, like everything else in this trainwreck of a language
            if hasattr(script, "NAISCRIPTNAME"): return script

    def get_args():
        if hasattr(p, "per_script_args"):
            args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
        elif hasattr(p, "script_args"):
            args = p.script_args[script.args_from:script.args_to]
        return args

    script = FindScript(p.scripts)
    global POSTPROCESS
    global running_scripts
    is_post =False
    if script is None and len(running_scripts) >= POSTPROCESS:
        script = running_scripts[POSTPROCESS-1]   
        if script.do_nai_post: is_post=True
        else: script=None        
    if script is None or hasattr(p, "NAI_enable") and not p.NAI_enable: 
        return modules.processing.process_images_pre_patch_4_nai(p)
        
    args = get_args()
    
    
    if args is None: return modules.processing.process_images_pre_patch_4_nai(p)
    if not is_post:
        script.initialize()        
        script.patched_process(p, *args)        
        
        if script.failed: raise Exception(script.failure)
        if script.disabled: return modules.processing.process_images_pre_patch_4_nai(p)
    p.nai_processed=None
    originalprocess = p.scripts.process
    def process_patched(self,p, **kwargs):
        nonlocal originalprocess
        originalprocess(p, **kwargs)
        p.scripts.process = originalprocess
        originalprocess=None
        
        if is_post:
            print(f'isinstance(p, StableDiffusionProcessingImg2Img) {isinstance(p, StableDiffusionProcessingImg2Img)}')
            if isinstance(p, StableDiffusionProcessingImg2Img) or hasattr(p, "init_images"):
                if hasattr(script,"post_process_i2i"):
                    r = script.post_process_i2i(p,*args)
                    if r is not None:
                        p.nai_processed = r
                        raise NAIGenException
        else:
            script.process_inner(p, *args) # Sets p.nai_processed if images have been processed

        if script.failed: raise Exception(script.failure)
        if hasattr(p,"nai_processed") and p.nai_processed is not None:
            # Make sure post process is called, at least one extension uses it to clean up from things done in process. 
            # This may also cause problems since batch process isn't called, but this seems less likely.
            global POSTPROCESS
            global running_scripts
            cpost = POSTPROCESS
            
            if len(running_scripts) <= cpost:
                running_scripts.append(script)
            else: running_scripts[cpost]=script
            
            POSTPROCESS += 1

            try:
                r = p.nai_processed
                for i in range(len(r.images)):
                    p.iteration = int( i/p.n_iter)
                    p.batch_index = i % p.batch_size
                
                    image = r.images[i]
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image
                    r.images[i] =image

                p.scripts.postprocess(p, p.nai_processed) 
            finally:
                POSTPROCESS -= 1
            
            raise NAIGenException
                
    try:
        p.scripts.process= process_patched.__get__(p.scripts, scripts.ScriptRunner)
        results = modules.processing.process_images_pre_patch_4_nai(p)        
        if originalprocess is not None:p.scripts.process=originalprocess
        
        if getattr(script,'include_nai_init_images_in_results',False):
            results.all_subseeds += script.all_subseeds 
            results.all_seeds += script.all_seeds 
            results.all_prompts += script.all_prompts 
            results.all_negative_prompts += script.all_negative_prompts 
            results.images += script.images
            results.infotexts += script.texts
        return results
            
    except NAIGenException:
        return p.nai_processed
    finally: 
        if originalprocess is not None:p.scripts.process=originalprocess
        

if not hasattr(modules.processing, 'process_images_pre_patch_4_nai'):
    modules.processing.process_images_pre_patch_4_nai = modules.processing.process_images_inner     
    print("Patching Image Processing for NAI Generator Script.")
modules.processing.process_images_inner = process_images_patched


# def on_script_unloaded():
    # if hasattr(modules.processing, 'process_images_pre_patch_4_nai'):
        # modules.processing.process_images = modules.processing.process_images_pre_patch_4_nai 

# script_callbacks.on_script_unloaded(on_script_unloaded)

