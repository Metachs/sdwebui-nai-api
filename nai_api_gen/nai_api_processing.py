from modules import scripts, script_callbacks, shared 
import modules 
from nai_api_gen import nai_api
from nai_api_gen import nai_stealth_text_info
import modules.processing
import modules.images as images

from nai_api_gen.nai_api_settings import DEBUG_LOG



PERMANT_PATCH = not hasattr(scripts.Script, 'before_process')
PATCHED = False

def script_setup():
    if PERMANT_PATCH:
        if not hasattr(modules.processing, 'process_images_pre_patch_4_nai'):
            modules.processing.process_images_pre_patch_4_nai = modules.processing.process_images_inner     
            print("WARNING: Out of date sd-webui detected. Permanently Patching Image Processing for NAI Generator Script. This may not work depending on the extensions present, if anything stops working, disable all other extensions and try again.")
        modules.processing.process_images_inner = process_images_patched
        script_callbacks.on_script_unloaded(unload)

def unload():
    if PERMANT_PATCH:
        modules.processing.process_images_inner = modules.processing.process_images_pre_patch_4_nai
        del modules.processing.process_images_pre_patch_4_nai

class NAIGenException(Exception):
    pass
    
def post_process_images(p,script,is_post):
    try:            
        script.in_post_process=True
        r = p.nai_processed
        plen=min(len(r.all_prompts), len( r.images))
        DEBUG_LOG("Number of Images: " ,plen)
        for i in range(plen):
            p.iteration = int( i/p.n_iter)
            p.batch_index = i % p.batch_size
            image = r.images[i]
            existing_info = image.info.copy()
            DEBUG_LOG(not is_post and shared.opts.samples_save and not p.do_not_save_samples, is_post, shared.opts.samples_save,p.do_not_save_samples)
            save_images = not is_post and shared.opts.samples_save and not p.do_not_save_samples                    
            pp = scripts.PostprocessImageArgs(image)
            p.scripts.postprocess_image(p, pp)                    
            if image != pp.image: 
                if shared.opts.data.get("nai_api_save_original", True):
                    images.save_image(image, p.outpath_samples, "", r.all_seeds[i], r.all_prompts[i], shared.opts.samples_format, info=r.infotexts[i], p=p,existing_info=existing_info)
                r.images.append(image)
                r.images[i]=pp.image
                r.infotexts.append(r.infotexts[i])
                r.all_prompts.append(r.all_prompts[i])
                r.all_negative_prompts.append(r.all_negative_prompts[i])
                r.all_seeds.append(r.all_seeds[i])
                r.all_subseeds.append(r.all_subseeds[i])
            if save_images:
                DEBUG_LOG ("Save Image: " ,i)                
                images.save_image(r.images[i], p.outpath_samples, "", r.all_seeds[i], r.all_prompts[i], shared.opts.samples_format, info=r.infotexts[i], p=p,existing_info=existing_info)
        p.scripts.postprocess(p, p.nai_processed) 
    finally:                
        script.in_post_process=False        


def process_images_patched(p):
    unpatch_pi()
    
    def FindScript(p):
        if p.scripts is None:
            return None
        for script in p.scripts.alwayson_scripts:  
            if hasattr(script, "NAISCRIPTNAME"):
                return script
                
    script = FindScript(p)
    is_post = False
    if script is not None and getattr(script,"in_post_process",False):
        if script.do_nai_post: is_post=True
        else: script=None
    if script is None or hasattr(p, "NAI_enable") and not p.NAI_enable: 
        return modules.processing.process_images_pre_patch_4_nai(p)
        
    if hasattr(p, "per_script_args"): args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
    elif hasattr(p, "script_args"): args = p.script_args[script.args_from:script.args_to]
    else: args = None
    
    if args is None: return modules.processing.process_images_pre_patch_4_nai(p)
    
    script.initialize()        
    script.nai_configuration(p, *args)
    
    if script.failed: raise Exception(script.failure)
    if script.disabled: return modules.processing.process_images_pre_patch_4_nai(p)

    p.nai_processed=None
    originalprocess = p.scripts.process
    
    def process_patched(self,p, **kwargs):
        nonlocal originalprocess
        originalprocess(p, **kwargs)
        p.scripts.process = originalprocess
        originalprocess=None
        script.nai_image_processsing(p, *args) # Sets p.nai_processed if images have been processed
        if script.failed: raise Exception(script.failure)
        if hasattr(p,"nai_processed") and p.nai_processed is not None:       
            post_process_images(p,script,is_post) 
            raise NAIGenException
    try:
        ori_ad = ad_add_whitelist(script) if script.do_nai_post else None
        p.scripts.process = process_patched.__get__(p.scripts, scripts.ScriptRunner)
        results = modules.processing.process_images_pre_patch_4_nai(p)        
        if originalprocess is not None: 
            p.scripts.process=originalprocess
            originalprocess=None
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
    except Exception as e:
        print(e) #TODO: See if there is a better way to log exceptions
        raise
    finally: 
        ad_rem_whitelist(ori_ad)        
        if originalprocess is not None: p.scripts.process=originalprocess
        
def patch_pi():
    global PATCHED
    global PERMANT_PATCH
    if PERMANT_PATCH or PATCHED: return
    PATCHED=True
    if modules.processing.process_images_inner == process_images_patched:        
        print ("Warning: process_images_inner already patched")
    else:        
        modules.processing.process_images_pre_patch_4_nai = modules.processing.process_images_inner     
        DEBUG_LOG("Patching Image Processing for NAI Generator Script.")
        modules.processing.process_images_inner = process_images_patched
        
def unpatch_pi():
    global PATCHED
    global PERMANT_PATCH
    if PERMANT_PATCH or not PATCHED: return
    PATCHED=False
    if (process_images_patched != modules.processing.process_images_inner):
        print ("ERROR: process_images_inner Not patched!")
    else:            
        DEBUG_LOG("Unpatching Image Processing for NAI Generator Script.")
        modules.processing.process_images_inner = modules.processing.process_images_pre_patch_4_nai

def ad_add_whitelist(script):
    o = "ad_script_names"    
    from pathlib import Path
    name = Path(script.filename).stem.strip()
    if o not in shared.opts.data: return None
    if name not in shared.opts.data[o]:
        ori = shared.opts.data[o]
        shared.opts.data[o] = f'{name},{ori}'
        return ori
    return None

def ad_rem_whitelist(ori):
    o = "ad_script_names"    
    if ori is None or o not in shared.opts.data: return
    shared.opts.data[o] = ori
    

# def on_script_unloaded():
    # if hasattr(modules.processing, 'process_images_pre_patch_4_nai'):
        # modules.processing.process_images = modules.processing.process_images_pre_patch_4_nai 

# script_callbacks.on_script_unloaded(on_script_unloaded)

