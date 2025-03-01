import os
import hashlib
import math
import re
import random
import time
import gradio as gr

from PIL import Image, ImageFilter, ImageOps,ImageChops
import numpy as np

import modules 
import modules.processing
import modules.images as images

from modules import scripts, script_callbacks, shared, sd_samplers, masking, devices,errors
from modules.processing import Processed, StableDiffusionProcessingTxt2Img,StableDiffusionProcessingImg2Img,create_infotext,apply_overlay

from nai_api_gen import nai_api
from nai_api_gen import nai_api_processing

from modules import scripts, script_callbacks, shared, extra_networks, ui

from modules.ui_components import ToolButton

import modules.images as images
from modules.processing import process_images
from modules import masking
import numpy as np
import cv2

from nai_api_gen.nai_api_settings import DEBUG_LOG,get_recommended_schedule

PREFIX = 'NAI'
do_local_img2img_modes = ["One Pass: NAI Img2Img/Inpaint", "Two Pass: NAI Txt2Img > Local Img2Img (Ignore Source Image)","Two Pass: NAI Img2Img/Inpaint > Local Img2Img"]
inpaint_mode_default = "Infill (No Denoise Strength)"
inpaint_mode_choices= [inpaint_mode_default,"Img2Img (Use Denoise Strength)" ]

class NAIGENScriptBase(scripts.Script):

    def __init__(self):    
        super().__init__()
        self.NAISCRIPTNAME = "NAI"    
        self.images = []
        self.texts = []
        self.fragments = []
        self.fragment_texts = []
        self.sampler_name = None
        self.api_connected = False
        self.disabled=False
        self.failed=False
        self.failure=""
        self.running=True
        self.do_nai_post=False        
        self.in_post_process=False
        self.width = 0
        self.height = 0
        self.mask = None
        self.cfg = 0
        self.steps = 0
        self.strength = 0
        self.vibe_count = shared.opts.data.get('nai_api_vibe_count',4)
        self.vibe_field_count = self.vibe_count*3
        self.last_request_time = 0
        
        self.reference_image = None


    def title(self):
        return self.NAISCRIPTNAME

    def show(self, is_img2img):
        return False
        
    def before_process(self, p, enable, *args):
        self.disabled = not getattr(p, 'NAI_enable', enable)
        if self.disabled: return
        nai_api_processing.patch_pi()
        
    def postprocess(self, p, enable,*args):
        if self.disabled: return
        nai_api_processing.unpatch_pi()
 
    def ui(self, is_img2img):
        inpaint_label = "NAI Inpainting"
        elempfx = 'img2img'if is_img2img else 'txt2img'
        with gr.Accordion(label=self.NAISCRIPTNAME, open=False):
            with gr.Row(variant="compact"):
                enable = gr.Checkbox(value=False, label="Enable")
                hr = gr.HTML()
                refresh = ToolButton(ui.refresh_symbol)
                refresh.click(fn= lambda: self.connect_api(), inputs=[], outputs=[enable,hr])
            with gr.Row(variant="compact", visible = is_img2img):
                do_local_img2img = gr.Dropdown(value=do_local_img2img_modes[0],choices= do_local_img2img_modes,type="index", label="Mode")
                if is_img2img:
                    inpaint_mode = gr.Dropdown(value=inpaint_mode_default, label=inpaint_label , choices=inpaint_mode_choices, type="index")
            with gr.Row(variant="compact"):
                model = gr.Dropdown(label= "Model",value=nai_api.nai_models[0],choices=nai_api.nai_models,type="value",show_label=True)
                sampler = gr.Dropdown(label="Sampler",value="Auto",choices=["Auto",*nai_api.NAI_SAMPLERS],type="value",show_label=True)
            with gr.Row(variant="compact"):
                noise_schedule = gr.Dropdown(label="Schedule",value="recommended",choices=["recommended","exponential","polyexponential","karras","native"],type="value")
                smea = gr.Radio(label="SMEA",value="Off",choices=["SMEA","DYN","Off"],type="value",show_label=True)
            with gr.Row(variant="compact"):
                dynamic_thresholding = gr.Checkbox(value=False, label='Decrisper',min_width=64)
                variety = gr.Checkbox(value=False, label='Variety+',min_width=64, elem_id = f"{elempfx}_nai_variety")
            with gr.Row(variant="compact",visible = is_img2img):
                extra_noise=gr.Slider(minimum=0.0, maximum=1.0 ,step=0.01, label='Noise', value=0.0)
                add_original_image = gr.Checkbox(value=True, label='Inpaint: Overlay Image')            
            with gr.Row(visible = False) as charmsg:
                gr.Markdown(
                    """<details>
                    <summary><strong>NAIv4 Syntax</strong></summary>
                    Use "CHAR:" at the start of a new line for each character prompt (not case sensitive).<br/>
                    The character prompt ends at the next line break, and can be embedded in the prompt (eg for styles).<br/>
                    Use "CHAR:POS:" to specify position using A1 - E5 (eg "CHAR:A1:", "CHAR:C3:") (Main Prompt only)<br/>
                    Use "CHAR:x,y:" to specify exact position, range is 0 to 1.0 (eg "CHAR:0.1,0.1:", "CHAR:0.5,0.5:") (Main Prompt only)<br/>                    
                    Can specify negatives in the negative prompt, will pair with main prompt char by order unless specified<br/>
                    Use "CHAR:1-9:" to manually match chars in negatives to main prompt (eg "CHAR:2:" matches the 2nd char)<br/>
                    '|' syntax for Characters is NOT supported, due to conflicts with other extensions<br/>
                    <br/>
                    If " Text:" is used inside the prompt, the rest of the prompt (including added styles) will be treated as a text tag<br/>
                    Use "Text:" at the start of a new line to define a text tag that ends at the next line break to prevent this.<br/>
                    <br/>
                    By default, sdwebui treats anything after '#' as a comment, which breaks NAI's action tag syntax<br/>
                    Use the configurable Alternate Action Tag ('::' by default, eg 'target::pointing') or disable "Stable Diffusion > Enable Comments"<br/>
                    <br/>
                    Prompt example:<br/>
                    &emsp;2girls<br/>
                    &emsp;CHAR:black hair<br/>
                    &emsp;CHAR:blonde hair<br/>
                    Negative Prompt:<br/>
                    &emsp;CHAR:<br/>
                    &emsp;CHAR:blue eyes<br/>
                    <br/>
                    Advanced:<br/>
                    &emsp;2girls<br/>
                    &emsp;CHAR:B3:black hair<br/>
                    &emsp;CHAR:D3:blonde hair<br/>
                    Negative Prompt:<br/>    
                    &emsp;CHAR:2:blue eyes<br/>
                    </details>""",
                    dangerously_allow_html=True
                )
            with gr.Accordion(label="Advanced", open=False):
                with gr.Row(variant="compact"):
                    cfg_rescale=gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='CFG Rescale', value=0.0)
                    # uncond_scale=gr.Slider(minimum=0.0, maximum=1.5, step=0.01, label='Uncond Scale', value=1.0,visible=False)
                skip_cfg_above_sigma=gr.Slider(minimum=0.0, maximum=100.0, step=1.0, label='skip_cfg_above_sigma (Manual Variety+ On=19 @1216x832)', value=0.0 ,visible=True, elem_id = f"{elempfx}_nai_cfg_skip")
                if not is_img2img:
                    inpaint_mode = gr.Dropdown(value=inpaint_mode_default, label=inpaint_label , choices=inpaint_mode_choices, type="index")                    
            def dovibefields(idx):
                with gr.Row(variant="compact"):
                    reference_image = gr.Image(label=f"Reference Image {idx}", elem_id=f"reference_image_{idx}", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA",min_width=164)
                    with gr.Column(min_width=64):
                        reference_information_extracted=gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label=f'Info Extracted {idx}', value=1.0,min_width=64)
                        reference_strength=gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, label=f'Reference Strength {idx}', value=0.6,min_width=64)

                reference_image_text=gr.Textbox(label=f"Reference Image Text Data {idx}", visible = False, value = "")

                reference_image.change( fn=None, _js= nai_api.vibe_processing_js, inputs=[reference_image], show_progress=False,outputs = [reference_image_text])
            
                return reference_image,reference_information_extracted,reference_strength,reference_image_text

            vibe_fields = []
            
            def dovibe(idx):
                with gr.Accordion(label=f'Vibe Transfer {idx+1}', open=False): 
                    fields = dovibefields(idx+1)
                    for fi in range(len(fields)):
                        if fi >= len(vibe_fields): vibe_fields.append([])
                        vibe_fields[fi].append(fields[fi])
                    if idx+1<self.vibe_count: dovibe(idx+1)
                    
            dovibe(0)
            with gr.Accordion(label='Director Tools', elem_id = f"nai_aug_{elempfx}", open=False , visible = is_img2img):                   
                with gr.Row(variant="compact", elem_id = f"nai_aug1_{elempfx}"):
                    augment_mode = gr.Dropdown(label="Augment Mode ",value="None",choices=["None",'recolorize',*nai_api.augment_modes])
                    defry=gr.Slider(minimum=0, maximum=5, step=1, label='Defry', value=0,min_width=64,visible=False)
                    emotion = gr.Dropdown(label="Emotion ",value=nai_api.augment_emotions[0],choices=[*nai_api.augment_emotions, "Prompt"],visible=False)
                with gr.Accordion(label='Levels',elem_id = f"nai_aug2_{elempfx}",visible = False, open=False) as levrow:
                    with gr.Row(variant="compact", elem_id = f"nai_aug21_{elempfx}"):
                        reclrLvlLo=gr.Slider(minimum=0, maximum=255, step=1, label='Black', value=0,min_width=64)
                        reclrLvlMid=gr.Slider(minimum=0, maximum=255, step=1, label='Gray', value=128,min_width=64)
                        reclrLvlHi=gr.Slider(minimum=0, maximum=255, step=1, label='White', value=255,min_width=64)
                    with gr.Row(variant="compact", elem_id = f"nai_aug22_{elempfx}"):
                        reclrLvlLoOut=gr.Slider(minimum=0, maximum=255, step=1, label='Out:Black', value=0,min_width=64)
                        reclrLvlHiOut=gr.Slider(minimum=0, maximum=255, step=1, label='Out:White', value=255,min_width=64)
                bgwarn = gr.HTML(value=f'<p>Disable Opus Free Gen Limit to use bg-removal. Background removal always has a cost.</p>',visible=False)
                augwarn = gr.HTML(value=f'<p>Prompt must not exceed 75 tokens or else generation will fail with a 500 Error.</p>',visible=False)
            def augchange(m,cl):
                isdfp = m in ['emotion','colorize','recolorize']
                return [ gr.update(visible = isdfp), gr.update(visible = m == 'emotion'), gr.update(visible = isdfp) , gr.update(visible = m =='recolorize'), gr.update(visible = cl and m == 'bg-removal') ]

            with gr.Accordion(label='Local Second Pass Overrides: Ignored if 0', open=False , visible = is_img2img):
                with gr.Row(variant="compact"):
                    nai_resolution_scale=gr.Slider(minimum=0.0, maximum=4.0, step=0.05, label='Scale', value=1.0)
                    nai_cfg=gr.Slider(minimum=0.0, maximum=30, step=0.05, label='CFG', value=0.0)
                    nai_steps=gr.Slider(minimum=0, maximum=150, step=1, label='Steps', value=0)
                    nai_denoise_strength=gr.Slider(minimum=0.0, maximum=1.0,step=0.01, label='Denoise strength', value=0.0)
                    keep_mask_for_local = gr.Checkbox(value=False, label="Keep inpaint mask for both passes")
            with gr.Accordion(label="Options", open=False):
                with gr.Row(variant="compact"):
                    qualityToggle = gr.Radio(value="Off", label="Quality Preset",choices=["Off","On"],type="index") 
                    ucPreset = gr.Radio(label="Negative Preset",value="None",choices=["Heavy","Light","Human","None"],type="index")
                    convert_prompts = gr.Dropdown(label="Convert Prompts for NAI ",value="Auto",choices=["Auto","Never","Always"])
                    cost_limiter = gr.Checkbox(value=True, label="Force Opus Free Gen Size/Step Limit")
                    nai_post = gr.Checkbox(value=True, label="Use NAI for Inpainting with ADetailer")
                    disable_smea_in_post = gr.Checkbox(value=True, label="Disable SMEA for Post/Adetailer")
                    legacy_v3_extend = gr.Checkbox(value=False, label="Legacy v3 (Disable Token Fix)")
                    
                    deliberate_euler_ancestral_bug = gr.Dropdown(value="Default", label="Deliberate EulerA Bug",choices=["Default","True","False"])
                    prefer_brownian = gr.Dropdown(value="Default", label="Prefer Brownian",choices=["Default","True","False"])
                    
        def on_enable(e,h):
            if e and not self.api_connected: return self.connect_api()
            return e,h
            
        if not self.skip_checks():
            enable.change(fn=on_enable, inputs=[enable,hr], outputs=[enable,hr])
          
        augment_mode.change(fn = augchange, inputs = [augment_mode,cost_limiter], outputs=[defry, emotion, augwarn,levrow,bgwarn], show_progress=False)
        
        model.change (fn = lambda mod: gr.update(visible= '4' in mod), inputs = [model], outputs=[charmsg] )

        # model.change(fn = lambda mod,msg : [gr.update(), gr.update(visible= '4' in mod)], inputs = [model], outputs=[model,charmsg] , show_progress=False)
        self.infotext_fields = [
            (enable, f'{PREFIX} enable'),
            (sampler, f'{PREFIX} sampler'),
            (noise_schedule, f'{PREFIX} noise_schedule'),
            (dynamic_thresholding, f'{PREFIX} dynamic_thresholding'),
            (model, f'{PREFIX} '+ 'model'),
            (smea, f'{PREFIX} '+ 'smea'),
            (skip_cfg_above_sigma, f'{PREFIX} '+ 'skip_cfg_above_sigma'),
            (cfg_rescale, f'{PREFIX} '+ 'cfg_rescale'),
            
            (nai_denoise_strength, f'{PREFIX} '+ 'nai_denoise_strength'),
            (nai_steps, f'{PREFIX} '+ 'nai_steps'),
            (nai_cfg, f'{PREFIX} '+ 'nai_cfg'),
            (extra_noise, f'{PREFIX} '+ 'extra_noise'),
            (legacy_v3_extend, f'{PREFIX} '+ 'legacy_v3_extend'),
            (add_original_image, f'{PREFIX} '+ 'add_original_image'),
            (augment_mode, f'{PREFIX} '+ 'augment_mode'),
            (defry, f'{PREFIX} '+ 'defry'),
            (emotion, f'{PREFIX} '+ 'emotion'),
            (reclrLvlLo, f'{PREFIX} '+ 'reclrLvlLo'),
            (reclrLvlHi, f'{PREFIX} '+ 'reclrLvlHi'),
            (reclrLvlMid, f'{PREFIX} '+ 'reclrLvlMid'),
            (reclrLvlLoOut, f'{PREFIX} '+ 'reclrLvlLoOut'),
            (reclrLvlHiOut, f'{PREFIX} '+ 'reclrLvlHiOut'),
            (variety, f'{PREFIX} '+ 'variety'),
            (deliberate_euler_ancestral_bug, f'{PREFIX} eulera_bug'),
            (prefer_brownian, f'{PREFIX} prefer_brownian'),
        ]

        def subvibeget(text,i,old_name):
            def key(i):
                return f'{PREFIX} {text} {i+1}'
            def keyold(i):
                if i==0: return f'{PREFIX} {old_name}'
                return f'{PREFIX} {old_name}{i+1}'
            def func(d):
                if key(i) in d: return d[key(i)]
                if keyold(i) in d: return d[keyold(i)]
                if i != 0 and f'reference_image_hash{i+1}' in d and key(0) in d:
                    return d[key(0)]
                return None
            return func

        def restorefunc(i):                        
            def restore(d,name,oldname):
                hash = d[name] if name in d else d[oldname] if oldname in d else None
                if not hash: return None
                imgp = os.path.join(shared.opts.outdir_init_images, f"{hash}.png")
                return imgp if os.path.exists(imgp) else None
            def func(x):
                return gr.update(value = restore(x, f'{PREFIX} Vibe Hash {i+1}',"reference_image_hash" if i==0 else f'reference_image_hash{i+1}'))
            return func

        for i in range(self.vibe_count):        
            if shared.opts.outdir_init_images and os.path.exists(shared.opts.outdir_init_images):
                self.infotext_fields.append((vibe_fields[0][i],restorefunc(i)))
            self.infotext_fields.append((vibe_fields[1][i],subvibeget('Vibe IE',i,'reference_information_extracted')))
            self.infotext_fields.append((vibe_fields[2][i],subvibeget('Vibe Strength',i,'reference_strength')))
        vibe_fields = [f for field in vibe_fields[::-1] for f in field[::-1]]
        self.vibe_field_count = len(vibe_fields)
        
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            if isinstance(field_name, str): self.paste_field_names.append(field_name)
            
        return [enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,add_original_image,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,deliberate_euler_ancestral_bug,prefer_brownian,keep_mask_for_local,*vibe_fields]

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
        status, opus, points, max = nai_api.subscription_status(key)
        return opus or points > 1   

    def subscription_status_message(self):
        key = self.get_api_key()
        status, opus, points, max = nai_api.subscription_status(key)
        DEBUG_LOG(f'Subscription Status: {status} {opus} {points} {max}')
        if status == -1: return False,"[ERROR] Missing API Key, enter in Settings > NAI API Generator"
        elif status == 401: return False,"Invalid API Key"
        elif status == 408: return False,"Request Timed out, try again."
        elif status != 200: return False,f"[API ERROR] Error Code: {status}"
        elif not opus and points <=1:
            return True, f'[API ERROR] Insufficient points! {points}'
        return True, f'[API OK] Anlas:{points} {"Opus" if opus else ""}'
    
    def setup_sampler_name(self,p, nai_sampler,noise_schedule='recommended'):
        noise_schedule_recommended = noise_schedule not in nai_api.noise_schedules
        if nai_sampler not in nai_api.NAI_SAMPLERS:
            sampler = sd_samplers.all_samplers_map.get(p.sampler_name)
            if sampler.name in ["DDIM"]: nai_sampler = "ddim_v3"
            else:            
                for ns in nai_api.NAI_SAMPLERS:
                    nsl = ns.lower().replace('ancestral','a')
                    if nsl in sampler.aliases or nsl +'_ka' in sampler.aliases:
                        nai_sampler = ns
                        if 'karras' in sampler.name.lower(): 
                            noise_schedule = 'karras'
                            noise_schedule_recommended = True
                        break
            if nai_sampler not in nai_api.NAI_SAMPLERS: 
                nai_sampler = shared.opts.data.get('NAI_gen_default_sampler', 'k_euler')
                p.sampler_name = sd_samplers.all_samplers_map.get(nai_sampler.replace('ancestral','a'),None) or p.sampler_name

        self.sampler_name = nai_sampler
        self.noise_schedule = noise_schedule
        self.noise_schedule_recommended = noise_schedule_recommended
        if self.noise_schedule.lower() not in nai_api.noise_schedules:        
            dic = {}
            for kv in shared.opts.nai_api_recommended_schedule_overrides.lower().split():
                if ':' in kv: 
                    k,v,*_ = kv.split(':')
                    dic[k]=v       
            self.noise_schedule = dic.get(self.sampler_name, 'karras')
            
    def initialize(self):
        DEBUG_LOG(f'initialize')        
        self.failed= False
        self.failure =""
        self.images=[]
        self.texts=[]
        self.fragments = []
        self.fragment_texts = []
        self.mask = None
        self.init_masked = None
        self.crop = None
        self.init_images = None
        self.has_protected_prompt = False
        
    def comment(self, p , c):
        print (c)
        if p is None or not hasattr(p, "comment"):return        
        p.comment(c)    
        
    def fail(self, p, c):
        self.comment(p,c)
        self.failed=True
        self.failure = c
        self.disabled=True
        shared.state.interrupt()
    
    def limit_costs(self, p, nai_batch = False, arbitrary_res = False):
        MAXSIZE = 1048576
        if p.width * p.height > MAXSIZE:
            scale = p.width/p.height
            p.height= int(math.sqrt(MAXSIZE/scale))
            p.width= int(p.height * scale)
            if not arbitrary_res:
                p.width = int(p.width/64)*64
                p.height = int(p.height/64)*64
            self.comment(p,f"Cost Limiter: Reduce dimensions to {p.width} x {p.height}")
        if nai_batch and p.batch_size > 1:
            p.n_iter *= p.batch_size
            p.batch_size = 1
            self.comment(p,f" Cost Limiter: Disable Batching")
        if p.steps > 28: 
            p.steps = 28
            self.comment(p,f"Cost Limiter: Reduce steps to {p.steps}")
    
    def adjust_resolution(self, p):
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
        
    def protect_prompts(self, p):
        def protect_prompt(s):
            l = s.rstrip().splitlines()
            if not l: return s
            l = l[-1]
            if ':' not in l: return s
            sp = l.split(':',maxsplit=2)
            tag = sp[0].strip().lower()
            if tag == "text" or tag == "char":
                self.has_protected_prompt = True
                return s.rstrip() + "\n::CHAREND::"
            return s
        p.prompt = protect_prompt(p.prompt)
        p.negative_prompt = protect_prompt(p.negative_prompt)
        
    def restore_prompts(self, p):
        if not self.has_protected_prompt: return
        def protect_prompt(s):
            return s.replace('::CHAREND::','')            
        if p.all_prompts: p.all_prompts = [protect_prompt(x) for x in p.all_prompts]
        if p.all_negative_prompts: p.all_negative_prompts = [protect_prompt(x) for x in p.all_negative_prompts]
        if getattr(p,'main_prompt',None): p.main_prompt = protect_prompt(p.main_prompt)
        if getattr(p,'main_negative_prompt',None): p.main_prompt = protect_prompt(p.main_negative_prompt)
        if p.prompt: p.prompt = protect_prompt(p.prompt)
        if p.negative_prompt: p.negative_prompt = protect_prompt(p.negative_prompt)
        
    def nai_configuration(self,p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,add_original_image,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,deliberate_euler_ancestral_bug,prefer_brownian,keep_mask_for_local,*args):
        if self.disabled: return        
        DEBUG_LOG(f'nai_configuration')   

        self.protect_prompts(p)
        
        self.do_nai_post=nai_post            
        self.cost_limiter = cost_limiter
        
        self.isimg2img = getattr(p, "init_images",None)
        DEBUG_LOG("self.isimg2img",self.isimg2img,"do_local_img2img",do_local_img2img)

        self.mask = getattr(p,"image_mask",None)        
        
        inpaint_mode =  getattr(p,f'{PREFIX}_'+ 'inpaint_mode',inpaint_mode)
        if isinstance(inpaint_mode,str):
            try: inpaint_mode = int(inpaint_mode) if len(inpaint_mode) == 1 else inpaint_mode_choices.index(inpaint_mode)
            except: inpaint_mode = 0
        self.inpaint_mode = inpaint_mode
        
        self.augment_mode =  getattr(p,f'{PREFIX}_'+ 'augment_mode',augment_mode)
        if self.isimg2img and self.augment_mode and (self.augment_mode in nai_api.augment_modes or self.augment_mode == 'recolorize') :
            if cost_limiter and self.augment_mode == 'bg-removal':
                self.comment(p,f"Cost Limiter: Background Removal is never free, switching to Declutter")
                self.augment_mode='declutter'       
            if cost_limiter: self.limit_costs(p, arbitrary_res = True)
                
        else: 
            self.augment_mode = ""
            if cost_limiter: self.limit_costs(p)
            self.adjust_resolution(p)
            self.setup_sampler_name(p, sampler,getattr(p,f'{PREFIX}_'+ 'noise_schedule',noise_schedule))

        if not self.isimg2img: do_local_img2img = 0
        elif isinstance(do_local_img2img,str):
            try: do_local_img2img = int(do_local_img2img) if len(do_local_img2img) == 1 else do_local_img2img_modes.index(do_local_img2img)
            except: do_local_img2img = 0
        self.do_local_img2img=do_local_img2img
        
        self.resize_mode = getattr(p,"resize_mode",0) 
        self.width = p.width
        self.height= p.height
        self.cfg = p.cfg_scale
        self.steps = p.steps
        self.strength = getattr(p,"denoising_strength",0)
        
        self.model = getattr(p,f'{PREFIX}_'+ 'model',model)
        
        if do_local_img2img== 1 or do_local_img2img == 2:
            self.set_local(p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,add_original_image,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,deliberate_euler_ancestral_bug,prefer_brownian,keep_mask_for_local)
        else:
            p.disable_extra_networks=True

        p.nai_processed=None
        
    def restore_local(self,p):
        p.width = self.width
        p.height= self.height
        p.cfg_scale = self.cfg
        p.steps = self.steps
        p.image_mask = self.mask
        p.denoising_strength = self.strength
        p.resize_mode = self.resize_mode
        
    def set_local(self,p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,add_original_image,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,deliberate_euler_ancestral_bug,prefer_brownian,keep_mask_for_local,*args):    
        if nai_resolution_scale> 0:
            p.width = int(p.width * nai_resolution_scale)
        p.height = int(p.height * nai_resolution_scale)
        if nai_cfg > 0: p.cfg_scale = nai_cfg
        if nai_steps > 0: p.steps = nai_steps
        if nai_denoise_strength > 0 and hasattr(p,"denoising_strength") : p.denoising_strength = nai_denoise_strength
        if not keep_mask_for_local and do_local_img2img == 2 and hasattr(p,"image_mask"): p.image_mask = None
        #Use standard resize mode if set to crop/fill resize mode. 
        if p.resize_mode == 1 or p.resize_mode == 2: 
            p.resize_mode = 0

    def save_init_img(self, image):
        hash = hashlib.md5(image.tobytes()).hexdigest()
        images.save_image(image.copy(), path=shared.opts.outdir_init_images, basename=None, forced_filename=hash, save_to_dirs=False)
        return hash

    def nai_preprocess(self,p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,add_original_image,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,deliberate_euler_ancestral_bug,prefer_brownian,keep_mask_for_local,*args):
        if self.disabled: return
        
        self.restore_prompts(p)        
        
        isimg2img=self.isimg2img
        do_local_img2img=self.do_local_img2img

        if do_local_img2img== 1 or do_local_img2img == 2: restore_local(p)
        
        if isimg2img and do_local_img2img == 0 and p.denoising_strength == 0:
            for i in range(len(p.init_images)):
                self.images.append(p.init_images[i])
                self.texts.append(self.infotext(p,i))
            p.nai_processed = Processed(p, self.images, p.seed, self.texts[0], subseed=p.all_subseeds[0], infotexts = self.texts) 
            return
        
        mask = None if not isimg2img or self.inpaint_mode == 2 else self.mask
        init_masked = None
        DEBUG_LOG("nai_preprocess")
        crop = None
        init_images=[]
        alt_masking = getattr(p,f'{PREFIX}_'+ 'alt_masking',False)
        self.alt_masking = alt_masking
        if isimg2img:
            DEBUG_LOG("init_images: " ,len(p.init_images))
            if mask is not None: 
                mask = mask.convert('L')
                DEBUG_LOG("Mask",mask.width, mask.height)                    
                if p.inpainting_mask_invert: mask = ImageOps.invert(mask)
                if not alt_masking:
                    if p.mask_blur > 0: mask = maskblur(mask, p.mask_blur)
                    self.mask_for_overlay = mask
                    if p.inpaint_full_res:
                        overlay_mask = mask
                        crop = masking.expand_crop_region(masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding), p.width, p.height, mask.width, mask.height)
                        mask = images.resize_image(2, mask.crop(crop), p.width, p.height)
                    else:
                        mask = images.resize_image(p.resize_mode if p.resize_mode < 3 else 0, mask, p.width, p.height)
                        overlay_mask = Image.fromarray(np.clip((np.array(mask).astype(np.float32)) * 2, 0, 255).astype(np.uint8))
                    mask = mask.convert('L')
                    min_mask = mask
                else: 
                    self.add_original_image=False
                    adjust_mask = getattr(p,f'{PREFIX}_'+ 'adjust_mask',0)                    
                    blur_outside_mask = getattr(p,f'{PREFIX}_'+ 'blur_outside_mask',False)
                    if p.inpaint_full_res:
                        crop = masking.expand_crop_region(masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding + p.mask_blur*2+expand*2), p.width, p.height, mask.width, mask.height)
                        mask = images.resize_image(2, mask.crop(crop), p.width, p.height)
                    else: mask = images.resize_image(p.resize_mode if p.resize_mode < 3 else 0, mask, p.width, p.height)
                    min_mask = mask.convert('L')
                    
                    if adjust_mask or 0 != 0: mask = dilate_contract(mask,adjust_mask)
                    if adjust_mask < 0: min_mask = mask.convert('L')
                    # if nai_mask_fill_blur > 8: min_mask = blur_out(min_mask, min(adjust_mask-8, adjust_mask//4))
                    mask_fill_blur = getattr(p,f'{PREFIX}_'+ 'mask_fill_blur',0)
                    if mask_fill_blur > 0:
                        min_mask = blur_out(min_mask, min(mask_fill_blur, adjust_mask))
                    
                    if p.mask_blur > 0: 
                        overlay_mask = maskblur(mask, p.mask_blur) if not blur_outside_mask else blur_out(mask, p.mask_blur)
                    else: overlay_mask = mask
                    
                    mask = mask.convert('L')
                    self.mask_for_overlay = overlay_mask

                init_masked=[]               

            for i in range(len(p.init_images)):
                image = p.init_images[i]
                if shared.opts.data.get('save_init_img',False) and not hasattr(p,'enable_hr'):
                    self.save_init_img(image)
                image = images.flatten(image, shared.opts.img2img_background_color)
                
                if not crop and (image.width != p.width or image.height != p.height):
                    image = images.resize_image(p.resize_mode if p.resize_mode < 3 else 0, image, p.width, p.height)
                    
                if init_masked is not None:
                    image_masked = Image.new('RGBa', (image.width, image.height))
                    DEBUG_LOG(image.width, image.height, p.width, p.height, mask.width, mask.height, overlay_mask.width, overlay_mask.height)
                    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(overlay_mask.convert('L')))
                    init_masked.append(image_masked.convert('RGBA'))
                
                if crop: image = images.resize_image(2, image.crop(crop), p.width, p.height)
 
                if mask and p.inpainting_fill != 1 and not self.augment_mode:
                    image.paste(masking.fill(image, min_mask).convert("RGBA").convert("RGBa"), mask=min_mask.convert('L'))

                init_images.append(image)
            
            self.mask = mask
            self.overlay = self.inpaint_mode == 1 or getattr(p,"inpaint_full_res",False) or alt_masking and add_original_image

            self.init_masked = init_masked
            self.crop = crop
            self.init_images = init_images
        else: 
            self.init_images = None
            self.mask = None
            self.overlay = False
            self.init_masked = None
            self.crop = None
            
        if self.augment_mode == 'recolorize':
            for i in range(len(self.init_images)):
                self.init_images[i] = nai_api.GrayLevels(self.init_images[i],reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut)
        
        # TODO: Skip this when augment is enabled.
        self.reference_image = []
        self.reference_information_extracted = []
        self.reference_strength = []
        
        for i in range(1,1+self.vibe_count):
        
            image = args[-(0 * self.vibe_count + i)]
            vie = getattr(p, f'{PREFIX}_VibeExtract_{i}', args[-(1 * self.vibe_count + i)])
            vstr = getattr(p, f'{PREFIX}_VibeStrength_{i}', args[-(2 * self.vibe_count + i)])            
            self.reference_information_extracted.append(vie)
            self.reference_strength.append(vstr)
            
            if hasattr(p,f'{PREFIX}_VibeOn_{i}') and not getattr(p,f'{PREFIX}_VibeOn_{i}'): image = None
            
            self.reference_image.append(image)
            if image is not None:
                if shared.opts.data.get('nai_api_vibe_pre', 'NAI') == 'NAI':
                    txt = args[-(3 * self.vibe_count + i)]
                    txt = txt.split(',',1) if isinstance(txt, str) else None 
                    if txt is not None and "image/png" in txt[0] and len(txt)>1:
                        self.reference_image[-1] = txt[1].strip()
                    else: print(f'Missing/Unknown Reference Image Data URL, sending Raw Image.')                
                elif shared.opts.data.get('nai_api_vibe_pre', 'NAI') == 'Scale' and max(image.width,image.height)>448:
                    size = 448
                    width = size if image.width>=image.height else size * image.width // image.height
                    height = size if image.height>=image.width else size * image.height // image.width                    
                    self.reference_image[-1] = image.resize((width, height),Image.Resampling.LANCZOS)

                p.extra_generation_params[f'{PREFIX} Vibe IE {i}' ] = self.reference_information_extracted[-1]
                p.extra_generation_params[f'{PREFIX} Vibe Strength {i}'] = self.reference_strength[-1]
                if shared.opts.data.get('save_init_img',False):
                    p.extra_generation_params[f'{PREFIX} Vibe Hash {i}'] = self.save_init_img(image)
        try:
            # Strip extra networks from prompt
            p.all_prompts, _ = extra_networks.parse_prompts(p.all_prompts)
            devices.torch_gc() # Cleanup after any GAN Resizing
        except Exception as e:
            # Don't fail when non-essential internal functions fail 
            errors.display(e, "Error During Cleanup", full_traceback= True)

    def nai_image_processsing(self,p, *args):
        self.nai_preprocess(p, *args)
        self.nai_generate_images(p, *args)
            
    def convert_to_nai(self, prompt, neg,convert_prompts="Always"):
        if convert_prompts != "Never":
            if convert_prompts == "Always" or nai_api.prompt_has_weight(prompt): prompt = nai_api.prompt_to_nai(prompt,True)
            if convert_prompts == "Always" or nai_api.prompt_has_weight(neg): neg = nai_api.prompt_to_nai(neg,True)

        prompt = prompt.replace('\\(','(').replace('\\)',')') # Un-Escape parenthesis
        neg = neg.replace('\\(','(').replace('\\)',')')
        
        return prompt, neg

    def parse_tags(self, prompt, neg):
        chars = []        
        def parse(original_prompt, negs):
            index = 0
            prompt = ""
            text_tag = None
            for l in original_prompt.splitlines():
                sp = l.split(':',maxsplit=2)                
                tag = sp[0].strip().lower()
                
                if len(sp) > 1 and tag == "text":
                    text_tag = ":".join(sp[1:])
                    continue
                    
                if len(sp)<2 or tag != "char": 
                    prompt = l if not prompt else f"{prompt}\n{l}"
                    continue
                    
                sp = sp[1:]
                xy=None
                if len(sp)>1:
                    val,txt = sp
                    try: 
                        if negs:
                            index = int(val)
                        elif val.count(',') == 1: 
                            xy = [float(v) for v in val.split(',')]
                        elif len(val.strip()) == 2:
                            coords = val.strip().lower()
                            xycoords = {'a':0.1,'b':0.3,'c':0.5,'d':0.7,'e':0.9,'1':0.1,'2':0.3,'3':0.5,'4':0.7,'5':0.9}
                            xy = xycoords[coords[0]], xycoords[coords[1]]
                        else:
                            txt = val + ':' + txt
                    except Exception:
                        txt = val + ':' + txt
                else:
                    txt = sp[0]
                    
                if shared.opts.data.get('nai_alt_action_tag', ''): txt = txt.replace(shared.opts.data.get('nai_alt_action_tag'), '#')
                
                nonlocal chars
                while index >= len(chars): chars += [{}]
                c = chars[index]
                c['uc' if negs else 'prompt'] = txt
                if xy is not None: c['center'] = {'x':xy[0],'y':xy[1]}
                index+=1
            return prompt if index > 0 else original_prompt , text_tag
            
        prompt, text_tag = parse(prompt,False)
        neg, _ = parse(neg,True)
        
        return prompt, neg, chars, text_tag
            
    def infotext(self,p,i):
        iteration = int(i / p.batch_size)
        batch = i % p.batch_size
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, iteration, batch)


    def nai_generate_images(self,p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,add_original_image,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,deliberate_euler_ancestral_bug,prefer_brownian,keep_mask_for_local,*args):
        if self.disabled or p.nai_processed is not None: return 
        DEBUG_LOG("nai_generate_images")
        DEBUG_LOG(enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,add_original_image,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,deliberate_euler_ancestral_bug,prefer_brownian,keep_mask_for_local)
        
        isimg2img=self.isimg2img
        do_local_img2img=self.do_local_img2img
        
        model = self.model
        smea = getattr(p,f'{PREFIX}_'+ 'smea',smea)
        if getattr(p,f'{PREFIX}_'+ 'sampler',None) in nai_api.NAI_SAMPLERS:
            self.sampler_name = getattr(p,f'{PREFIX}_'+ 'sampler',None)        
            sampler = self.sampler_name
        dynamic_thresholding = getattr(p,f'{PREFIX}_'+ 'dynamic_thresholding',dynamic_thresholding)
        skip_cfg_above_sigma = getattr(p,f'{PREFIX}_'+ 'skip_cfg_above_sigma',skip_cfg_above_sigma)
        variety = getattr(p,f'{PREFIX}_'+ 'variety',variety)
        cfg_rescale = getattr(p,f'{PREFIX}_'+ 'cfg_rescale',cfg_rescale)        
        extra_noise = getattr(p,f'{PREFIX}_'+ 'extra_noise',extra_noise)        
        add_original_image = getattr(p,f'{PREFIX}_'+ 'add_original_image',add_original_image)        
        extra_noise = max(getattr(p,"extra_noise",0) , extra_noise)
        
        defry = getattr(p,f'{PREFIX}_'+ 'defry',defry)        
        emotion = getattr(p,f'{PREFIX}_'+ 'emotion',emotion)
        inpaint_mode=self.inpaint_mode
        if self.inpaint_mode == 1 or self.alt_masking:
            add_original_image = False
        
        if disable_smea_in_post and self.in_post_process:            
            smea = "None" # Disable SMEA during post  
        
        p.extra_generation_params[f'{PREFIX} enable'] = True

        if self.augment_mode:
            p.extra_generation_params[f'{PREFIX} '+ 'augment_mode'] = augment_mode
            if self.augment_mode in ['colorize','emotion','recolorize']: 
                p.extra_generation_params[f'{PREFIX} '+ 'defry'] = defry
            if self.augment_mode in ['emotion']:             
                p.extra_generation_params[f'{PREFIX} '+ 'emotion'] = emotion
            if self.augment_mode in ['recolorize']:             
                p.extra_generation_params[f'{PREFIX} '+ 'reclrLvlLo'] = reclrLvlLo
                p.extra_generation_params[f'{PREFIX} '+ 'reclrLvlHi'] = reclrLvlHi
                p.extra_generation_params[f'{PREFIX} '+ 'reclrLvlMid'] = reclrLvlMid
                p.extra_generation_params[f'{PREFIX} '+ 'reclrLvlLoOut'] = reclrLvlLoOut
                p.extra_generation_params[f'{PREFIX} '+ 'reclrLvlHiOut'] = reclrLvlHiOut
        else:            
            p.extra_generation_params[f'{PREFIX} sampler'] = self.sampler_name if str(sampler).lower() in nai_api.NAI_SAMPLERS else sampler
            p.extra_generation_params[f'{PREFIX} noise_schedule'] = noise_schedule if self.noise_schedule_recommended else self.noise_schedule
            p.extra_generation_params[f'{PREFIX} dynamic_thresholding'] = dynamic_thresholding
            p.extra_generation_params[f'{PREFIX} '+ 'model'] = model
            p.extra_generation_params[f'{PREFIX} '+ 'smea'] = smea
            p.extra_generation_params[f'{PREFIX} '+ 'qualityToggle'] = qualityToggle
            p.extra_generation_params[f'{PREFIX} '+ 'ucPreset'] = ucPreset
            if skip_cfg_above_sigma:
                p.extra_generation_params[f'{PREFIX} '+ 'skip_cfg_above_sigma'] = skip_cfg_above_sigma
            elif variety:
                p.extra_generation_params[f'{PREFIX} '+ 'variety'] = variety
            p.extra_generation_params[f'{PREFIX} '+ 'cfg_rescale'] = cfg_rescale
            p.extra_generation_params[f'{PREFIX} '+ 'legacy_v3_extend'] = legacy_v3_extend
            
            if self.do_local_img2img != 0:        
                if nai_denoise_strength!= 0: p.extra_generation_params[f'{PREFIX} '+ 'nai_denoise_strength'] = nai_denoise_strength
                if nai_steps!= 0: p.extra_generation_params[f'{PREFIX} '+ 'nai_steps'] = nai_steps
                if nai_cfg!= 0: p.extra_generation_params[f'{PREFIX} '+ 'nai_cfg'] = nai_cfg
                if nai_resolution_scale!= 1: p.extra_generation_params[f'{PREFIX} '+ 'nai_resolution_scale'] = nai_resolution_scale
            if isimg2img:
                p.extra_generation_params[f'{PREFIX} '+ 'extra_noise'] = extra_noise                
            
            if prefer_brownian.lower() in ['true', 'false']:            
                prefer_brownian= prefer_brownian.lower() == "true"
                p.extra_generation_params[f'{PREFIX} prefer_brownian'] = prefer_brownian            
            else: prefer_brownian=None
            
            if deliberate_euler_ancestral_bug.lower() in ['true', 'false']:            
                deliberate_euler_ancestral_bug = deliberate_euler_ancestral_bug.lower() == "true"
                p.extra_generation_params[f'{PREFIX} eulera_bug'] = deliberate_euler_ancestral_bug            
            else: deliberate_euler_ancestral_bug=None
        
        def getparams(i,n_samples=1):
            seed =int(p.all_seeds[i])
            
            if getattr(p,'cfg_scaleses',None) and p.cfg_scaleses[i]: p.cfg_scale=p.cfg_scaleses[i]
            if getattr(p,'sampler_nameses',None) and p.sampler_nameses[i]:
                if f'{PREFIX} sampler' in p.extra_generation_params: p.extra_generation_params.pop(f'{PREFIX} sampler')
                if ':' in p.sampler_nameses[i]: 
                    if f'{PREFIX} noise_schedule' in p.extra_generation_params: p.extra_generation_params.pop(f'{PREFIX} noise_schedule')
                    self.setup_sampler_name(p,*p.sampler_nameses[i].split(':',1))
                else: self.setup_sampler_name(p,p.sampler_nameses[i])
            
            image= None if ( not isimg2img or self.do_local_img2img == 1 or self.init_images is None or len(self.init_images) == 0) else self.init_images[i % len(self.init_images)]
                
            prompt,neg = self.convert_to_nai(p.all_prompts[i],  p.all_negative_prompts[i], convert_prompts)
            
            mask = self.mask
            if '4' in model: 
                prompt,neg,chars,text_tag = self.parse_tags(prompt,neg)
                if mask and inpaint_mode!=1:
                    mask = mask.resize((self.mask.width//8, self.mask.height//8),Image.Resampling.NEAREST)
                    mask = mask.resize((self.mask.width, self.mask.height),Image.Resampling.NEAREST)
                    mask = clip(mask, 128)
                    mask = mask.convert("RGBA")
            else: 
                chars = []
                text_tag = None
            
            if self.augment_mode:
                return nai_api.AugmentParams('colorize' if self.augment_mode == 'recolorize' else self.augment_mode,image,p.width,p.height,prompt,defry,emotion,seed)
                
            return nai_api.NAIGenParams(prompt, neg, seed=seed , width=p.width, height=p.height, scale=p.cfg_scale, sampler = self.sampler_name, steps=p.steps, noise_schedule=self.noise_schedule,sm= "smea" in str(smea).lower(), sm_dyn="dyn" in str(smea).lower(), cfg_rescale=cfg_rescale,uncond_scale=0 ,dynamic_thresholding=dynamic_thresholding,model=model,qualityToggle = qualityToggle == 1, ucPreset = ucPreset , noise = extra_noise, image = image, strength= p.denoising_strength,extra_noise_seed = seed if p.subseed_strength <= 0 else int(p.all_subseeds[i]),overlay=add_original_image, mask = mask if inpaint_mode!=1 else None,legacy_v3_extend=legacy_v3_extend, reference_image=self.reference_image,reference_information_extracted=self.reference_information_extracted,reference_strength=self.reference_strength,n_samples=n_samples,variety=variety,skip_cfg_above_sigma=skip_cfg_above_sigma,deliberate_euler_ancestral_bug=deliberate_euler_ancestral_bug,prefer_brownian=prefer_brownian, characterPrompts=chars,text_tag=text_tag)
        
        while len(self.images) < p.n_iter * p.batch_size and not shared.state.interrupted:
            DEBUG_LOG("Loading Images: ",len(self.images) // p.batch_size,p.n_iter, p.batch_size)
            count = len(self.images)
            self.load_image_batch(p,getparams)            
            DEBUG_LOG("Read Image:",count)
            for i in range(count, len(self.images)):
                if self.images[i] is None:
                    self.images[i] = self.init_images[i] if self.init_images and i < len(self.init_images) else Image.new("RGBA",(p.width, p.height), color = "black")
                    self.images[i].is_transient_image = True
                elif isimg2img and getattr(p,"inpaint_full_res",False) and shared.opts.data.get('nai_api_save_fragments', False):
                    images.save_image(self.images[i], p.outpath_samples, "", p.all_seeds[i], p.all_prompts[i], shared.opts.samples_format, info=self.texts[i], suffix="-nai-init-image" if self.do_local_img2img != 0 else "")
        
        
        if self.do_local_img2img == 0:
           p.nai_processed = Processed(p, self.images, p.seed, self.texts[0] if len(self.texts) >0 else "", subseed=p.all_subseeds[0], infotexts = self.texts) 
        elif self.do_local_img2img== 1 or self.do_local_img2img == 2:
            self.all_seeds = p.all_seeds.copy()
            self.all_subseeds = p.all_subseeds.copy()
            self.all_prompts = p.all_prompts.copy()
            self.all_negative_prompts = p.all_negative_prompts.copy()
            
            self.set_local(p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,add_original_image,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,deliberate_euler_ancestral_bug,prefer_brownian,keep_mask_for_local)
                
            p.init_images = self.images.copy()
            self.include_nai_init_images_in_results=True
    
        self.images+=self.fragments
        self.texts+=self.fragment_texts
        
    def load_image_batch(self, p, getparams):    
        i = len(self.images)
        parameters = getparams(i)
        self.load_image(p,i, parameters)
        if shared.opts.live_previews_enable and self.images[i]!=None: 
            shared.state.assign_current_image(self.images[i].copy())
        if i == 0 and self.texts[i]:
            import modules.paths as paths
            with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    file.write(self.texts[i])

    def begin_request(self,p,i,parameters):    

        print(f"Requesting Image {i+1}/{p.n_iter*p.batch_size}: {p.width} x {p.height} - {p.steps} steps.")
        if shared.opts.data.get('nai_query_logging', False):                     
            print(re.sub("\"image\":\".*?\"","\"image\":\"\"" ,re.sub("\"mask\":\".*?\"","\"mask\":\"\"" ,re.sub("\"reference_image\":\".*?\"","\"reference_image\":\"\"" ,re.sub("\"reference_image_multiple\":\[.*?\]","\"reference_image_multiple\":\"\"" ,parameters)))))

        minimum_delay = shared.opts.data.get('nai_api_delay', 0)
        wait_time_rand = shared.opts.data.get('nai_api_delay_rand', 0)
        
        if minimum_delay + wait_time_rand > 0 and self.last_request_time is not None:         
            delay = ( minimum_delay + random.randint(0, wait_time_rand) )
            if delay - (time.time() - self.last_request_time) > 0: 
                print (f"Waiting {delay  - (time.time() - self.last_request_time):.2f}s before sending request")
                time.sleep(delay  - (time.time() - self.last_request_time))
                
        self.last_request_time = time.time()
        
    def request_complete(self,p,idx,*results):
        start = self.last_request_time
        self.last_request_time = time.time()
        has_error = False
        msg = f"Request Completed in {self.last_request_time - start:.2f}s"
        gentime = 0
        retry_count = 0
        for i, result in enumerate(results):   
            if result.status_code == 200:
                if result.files and isinstance(result.files[0], Image.Image):
                    gentime = max(gentime, nai_api.tryfloat(result.files[0].info.get('Generation time',0),0))
            else:
                err = f'Request {idx + i + 1}/{p.n_iter*p.batch_size}: {result.message}'                
                self.comment(p,err)
                if result.status_code < 0: errors.display(result, "NAI Image Request", full_traceback= True)
                has_error = True
            retry_count += result.retry_count
                
        if retry_count>0: msg = f'{msg} Retries: {retry_count:.2f}s'
        if gentime>0: msg = f'{msg} Generation time: {gentime:.2f}s'
        if gentime>0 and retry_count==0:
            if not hasattr(self,'times'):
                self.waits=[]
                self.times=[]
                self.rats=[]
                self.gents=[]
            self.rats.append((self.last_request_time - start -gentime)/(self.last_request_time - start))
            self.gents.append(gentime)
            self.waits.append(self.last_request_time - start -gentime)
            self.times.append(self.last_request_time - start)
            msg = f'{msg} Wait Time: {self.last_request_time - start -gentime:.2f}s'
            msg = f'{msg} Average Gen: {sum(self.gents)/len(self.gents):.2f}s'
            msg = f'{msg} Average Total: {sum(self.times)/len(self.times):.2f}s'
            msg = f'{msg} Average Wait: {sum(self.waits)/len(self.waits):.2f}s'
            msg = f'{msg} Ratio: {sum(self.rats)/len(self.rats):.2f}'
        if not has_error and idx == 0: self.comment(p,msg)
        return has_error
        
    def load_image(self, p, i, parameters, batch_size = 1):    
        key = self.get_api_key()        
        self.begin_request(p,i,parameters)
        result = nai_api.POST(key, parameters,timeout = nai_api.get_timeout(shared.opts.data.get('nai_api_timeout',90),p.width,p.height,p.steps), attempts = 1 if self.augment_mode else shared.opts.data.get('nai_api_retry', 2) , wait_on_429 = shared.opts.data.get('nai_api_wait_on_429', 30) , wait_on_429_time = shared.opts.data.get('nai_api_wait_on_429_time', 5),url = nai_api.NAI_AUGMENT_URL if self.augment_mode else nai_api.NAI_IMAGE_URL)
                
        if self.request_complete(p,i,result): 
            self.images.append(None)
            self.texts.append("")
            return            
        image = result.files
        
        for ii in range(len(image)):
            self.images.append(image[ii])
            self.texts.append(self.infotext(p,i+ii if ii < batch_size else i))
            self.process_result_image(p,i+ii if ii < batch_size else i)
            
        DEBUG_LOG("Read Image:",i)

    def process_result_image(self, p, i):
        def add_fragment(image = None):
            self.fragments.append(image or self.images[i])
            self.fragment_texts.append(self.texts[i])
        original = self.images[i]
        original.is_original_image = True
        if self.crop is not None:
            if shared.opts.data.get('nai_api_all_images', False): add_fragment()
            over = self.init_masked[i % len(self.init_masked)]
            image = Image.new("RGBA", over.size )
            image.paste(images.resize_image(1, self.images[i].convert("RGB"), self.crop[2]-self.crop[0], self.crop[3]-self.crop[1]), (self.crop[0], self.crop[1]))
            image.alpha_composite(over)
            image.original_nai_image = original
            self.images[i] = image
            if hasattr(self, 'mask_for_overlay') and self.mask_for_overlay and any([shared.opts.save_mask, shared.opts.save_mask_composite, shared.opts.return_mask, shared.opts.return_mask_composite]):
                if shared.opts.return_mask:
                    image_mask = self.mask_for_overlay.convert('RGB')
                    fragments.append(image_mask)
                    add_fragment(image_mask)

                if shared.opts.return_mask_composite:
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, self.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')
                    add_fragment(image_mask_composite)
                    fragments.append(image_mask_composite)
        elif self.mask is not None and self.overlay:
            if shared.opts.data.get('nai_api_all_images', False): add_fragment()
            image = self.images[i].convert("RGB").convert("RGBA")
            image.alpha_composite(self.init_masked[i % len(self.init_masked)])
            image = image.convert("RGB")
            image.original_nai_image = original
            self.images[i] = image

def maskblur(image, val):
    size = 2 * int(2.5 * val + 0.5) + 1
    ar = cv2.GaussianBlur(np.array(image), (size, size), val)
    return Image.fromarray(ar)
def clip(image,val):
    ar = np.array(image)
    ar = np.where(ar>=val,255,0)
    return Image.fromarray(ar).convert("L")
def expand_mask(mask,amount):
    amount = int(amount)
    oversrc = Image.new("L", mask.size, 0)
    expand = mask.copy()
    from PIL import ImageChops
    def past(x,y):
        nonlocal expand
        over= oversrc.copy()
        over.paste(mask,(x,y))
        expand = ImageChops.lighter(expand,over)        
    past(amount,0)
    past(amount,amount)
    past(0,amount)
    past(-amount,0)
    past(-amount,-amount)
    past(amount,-amount)
    past(-amount,amount)
    past(0,-amount)
    return expand    
def dilate_contract(image,amount):
    clpval=32 if amount >= 0 else 223
    image = clip(image,clpval)            
    amount = int(amount)
    shift = amount // 2
    oversrc = Image.new("L", image.size, 0 if shift>=0 else 255)
    expand = image.copy()
    def paste(x,y):
        nonlocal expand
        over= oversrc.copy()
        over.paste(image,(x,y))
        expand = ImageChops.lighter(expand,over) if shift >=0 else ImageChops.darker(expand,over)
    paste(shift,0)
    paste(shift,shift)
    paste(0,shift)
    paste(-shift,0)
    paste(-shift,-shift)
    paste(shift,-shift)
    paste(-shift,shift)
    paste(0,-shift)
    return clip(maskblur(expand,abs(amount) // 2),clpval)
def blur_out(image, amount):
    image = clip(image,32)
    amount = amount // 2
    b = maskblur(image, amount)
    b = ImageChops.lighter(image,b)
    b = clip(b, 1)
    b = maskblur(b, amount)            
    image = ImageChops.lighter(image,b)
    return image
    