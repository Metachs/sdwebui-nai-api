import os
import hashlib
import math
import re
import random
import json
import time
import gradio as gr
from io import BytesIO
import base64

from PIL import Image, ImageFilter, ImageOps,ImageChops, PngImagePlugin
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
inpaint_mode_default = "Inpaint Model"
inpaint_mode_choices= [inpaint_mode_default,"Img2Img" ]
base_dir = scripts.basedir()
elempfx = 'txt2img'

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
        self.vibe_count_v4 = shared.opts.data.get('nai_api_vibe_v4_count',4)
        self.vibe_field_count = self.vibe_count*3
        self.vibe_field_count_v4 = self.vibe_count_v4*4
        self.last_request_time = 0
        self.incurs_cost = False
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
        global elempfx
        elempfx = 'img2img'if is_img2img else 'txt2img'
        
        self.v3_only_ui = []
        self.v4_only_ui = []
        
        with gr.Accordion(label=self.NAISCRIPTNAME, open=False):
            with gr.Row(variant="compact"):
                enable = gr.Checkbox(value=False, label="Enable")
                hr = gr.HTML()
                refresh = ToolButton(ui.refresh_symbol)
                refresh.click(fn= lambda: self.connect_api(), inputs=[], outputs=[enable,hr])
            with gr.Row(visible = is_img2img):
                do_local_img2img = gr.Dropdown(value=do_local_img2img_modes[0],choices= do_local_img2img_modes,type="index", label="Mode")
                if is_img2img:
                    inpaint_mode = gr.Dropdown(value=inpaint_mode_default, label=inpaint_label , choices=inpaint_mode_choices, type="index")
            with gr.Row():
                model = gr.Dropdown(label= "Model",value=nai_api.nai_models[0],choices=nai_api.nai_models,type="value",show_label=True)
                sampler = gr.Dropdown(label="Sampler",value="Auto",choices=["Auto",*nai_api.NAI_SAMPLERS],type="value",show_label=True)
            with gr.Row():
                cfg_rescale=gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='CFG Rescale', value=0.0)
                noise_schedule = gr.Dropdown(label="Schedule",value="recommended",choices=["recommended","exponential","polyexponential","karras","native"],type="value")
            with gr.Row():
                variety = gr.Checkbox(value=False, label='Variety+', elem_id = f"{elempfx}_nai_variety",min_width=64)
            skip_cfg_above_sigma=gr.Slider(minimum=0.0, maximum=100.0, step=0.01, info='skip cfg above sigma (Manual Variety+)', value=0.0 ,show_label=False,visible=True, elem_id = f"{elempfx}_nai_cfg_skip",min_width=196)
                
            with gr.Row() as smdyn:
                dynamic_thresholding = gr.Checkbox(value=False, label='Decrisper',min_width=64)
                smea = gr.Radio(label="SMEA",value="Off",choices=["SMEA","DYN","Off"],type="value",show_label=True)
                self.v3_only_ui.append(smdyn)
            with gr.Row(visible = is_img2img):
                extra_noise=gr.Slider(minimum=0.0, maximum=1.0 ,step=0.01, label='Noise', value=0.0)
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
                self.v4_only_ui.append(charmsg)
                
            # with gr.Accordion(label="Advanced", open=False):
                # with gr.Row(variant="compact"):                    
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
                with gr.Accordion(label=f'V3 Vibe Transfer {idx+1}', open=False) : 
                    fields = dovibefields(idx+1)
                    for fi in range(len(fields)):
                        if fi >= len(vibe_fields): vibe_fields.append([])
                        vibe_fields[fi].append(fields[fi])
                    if idx+1<self.vibe_count: dovibe(idx+1)
                    
            with gr.Group(elem_id = f"nai_vb3_{elempfx}", visible = False) as vibes_v3:
                dovibe(0)

            self.v3_only_ui.append(vibes_v3)
            
            vibe_fields4,vibes_v4, normalize_reference_strength_multiple,normalize_negatives,normalize_level, *_ = self.vibe_v4_ui(model,vibe_fields)

            with gr.Accordion(label='Character Reference', elem_id = f"nai_cref_{elempfx}", open=False):
                crwarn = gr.HTML(value=f'<p>Always costs 5 tokens, Opus Free Gen Limit must be disabled to use.</p>',visible=True)
                with gr.Row(variant="compact", elem_id = f"nai_cref1_{elempfx}"):
                    cref_image_file = gr.Image(label=f"Source Image", elem_id=f"cref_image_{elempfx}", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA")                       
                    with gr.Column(variant="compact", elem_id = f"nai_cref1_{elempfx}"):
                        cref_style = gr.Checkbox(value=True, elem_id=f"cref_style_{elempfx}", label="Style Aware")    
                        cref_fidel = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, elem_id=f"cref_fidel_{elempfx}", label="Fidelity")    
                cref_image=gr.Textbox(label=f"Character Reference Image Text Data", visible = False, value = "")
                cref_image_file.change( fn=None, _js= nai_api.cr_processing_js, inputs=[cref_image_file], show_progress=False,outputs = [cref_image])
            
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
                        reclrLvlAlpha=gr.Slider(minimum=0, maximum=1, step=0.01, label='Alpha', value=1.0,min_width=64)
                bgwarn = gr.HTML(value=f'<p>Disable Opus Free Gen Limit to use bg-removal. Background removal always has a cost.</p>',visible=False)
                augwarn = gr.HTML(value=f'<p>Prompt must not exceed 75 tokens or else generation will fail with a 500 Error.</p>',visible=False)
            def augchange(m,cl):
                isdfp = m in ['emotion','colorize','recolorize']
                return [ gr.update(visible = isdfp), gr.update(visible = m == 'emotion'), gr.update(visible = isdfp) , gr.update(visible = m =='recolorize'), gr.update(visible = cl and m == 'bg-removal') ]

            with gr.Accordion(label='Local Second Pass Overrides: Ignored if 0', open=False , visible = is_img2img):
                with gr.Row():
                    nai_resolution_scale=gr.Slider(minimum=0.0, maximum=4.0, step=0.05, label='Scale', value=1.0)
                    nai_cfg=gr.Slider(minimum=0.0, maximum=30, step=0.05, label='CFG', value=0.0)
                    nai_steps=gr.Slider(minimum=0, maximum=150, step=1, label='Steps', value=0)
                    nai_denoise_strength=gr.Slider(minimum=0.0, maximum=1.0,step=0.01, label='Denoise strength', value=0.0)
                    keep_mask_for_local = gr.Checkbox(value=False, label="Keep inpaint mask for both passes")
            with gr.Accordion(label="Options", open=False):
                cost_limiter = gr.Checkbox(value=True, label="Force Opus Free Gen Limits")
                with gr.Row():
                    qualityToggle = gr.Radio(value="Off", label="Quality Preset",choices=["Off","On"],type="index") 
                    convert_prompts = gr.Dropdown(label="Convert Prompts for NAI ",value="Auto",choices=["Auto","Never","Always"])                   
                with gr.Row():
                    ucPreset = gr.Radio(label="Negative Preset",value="None",choices=["None","Heavy","Light","Human","Furry"],type="value")
                    
                with gr.Row():
                    nai_post = gr.Checkbox(value=True, label="Use NAI for Inpainting with ADetailer")
                with gr.Row():
                    disable_smea_in_post = gr.Checkbox(value=True, label="Disable SMEA for Post/Adetailer")
                    self.v3_only_ui.append(disable_smea_in_post)
                    if not is_img2img:
                        inpaint_mode = gr.Dropdown(value=inpaint_mode_default, label=inpaint_label , choices=inpaint_mode_choices, type="index")
                with gr.Row():
                    legacy_uc = gr.Checkbox(value=False, label="V4 Legacy Conditioning")
                    self.v4_only_ui.append(legacy_uc)
                    
                    legacy_v3_extend = gr.Checkbox(value=False, label="Legacy v3 (Disable Token Fix)")
                    self.v3_only_ui.append(legacy_v3_extend)
                    
                    deliberate_euler_ancestral_bug = gr.Dropdown(value="Default", label="Deliberate EulerA Bug",choices=["Default","True","False"])
                    prefer_brownian = gr.Dropdown(value="Default", label="Prefer Brownian",choices=["Default","True","False"])
                    
        def on_enable(e,h):
            if e and not self.api_connected: return self.connect_api()
            return e,h
            
        if not self.skip_checks():
            enable.change(fn=on_enable, inputs=[enable,hr], outputs=[enable,hr])
          
        augment_mode.change(fn = augchange, inputs = [augment_mode,cost_limiter], outputs=[defry, emotion, augwarn,levrow,bgwarn], show_progress=False)
        

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
            (legacy_uc, f'{PREFIX} '+ 'legacy_uc'),
            (augment_mode, f'{PREFIX} '+ 'augment_mode'),
            (defry, f'{PREFIX} '+ 'defry'),
            (emotion, f'{PREFIX} '+ 'emotion'),
            (reclrLvlLo, f'{PREFIX} '+ 'reclrLvlLo'),
            (reclrLvlHi, f'{PREFIX} '+ 'reclrLvlHi'),
            (reclrLvlMid, f'{PREFIX} '+ 'reclrLvlMid'),
            (reclrLvlLoOut, f'{PREFIX} '+ 'reclrLvlLoOut'),
            (reclrLvlHiOut, f'{PREFIX} '+ 'reclrLvlHiOut'),
            (reclrLvlAlpha, f'{PREFIX} '+ 'reclrLvlAlpha'),
            (variety, f'{PREFIX} '+ 'variety'),
            (deliberate_euler_ancestral_bug, f'{PREFIX} eulera_bug'),
            (prefer_brownian, f'{PREFIX} prefer_brownian'),
            (normalize_reference_strength_multiple, f'{PREFIX} normalize_reference_strength'),
            (normalize_negatives, f'{PREFIX} normalize_negatives'),
            (normalize_level, f'{PREFIX} normalize_level'),
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
        
        vibe_fields4 = self.vibe_v4_finalize(model, vibes_v4, vibe_fields4)
        
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            if isinstance(field_name, str): self.paste_field_names.append(field_name)
            
        return [enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,reclrLvlAlpha,deliberate_euler_ancestral_bug,prefer_brownian,legacy_uc,normalize_reference_strength_multiple,normalize_negatives,normalize_level,cref_image,cref_style,cref_fidel,keep_mask_for_local,*vibe_fields4,*vibe_fields]

    def vibe_v4_finalize(self,model,vibes_v4, vibe_fields4):
        v3_only_len = len(self.v3_only_ui)
        v4_only_len = len(self.v4_only_ui)
        
        model.change (fn = lambda mod: [gr.update(visible= '4' in mod or shared.opts.nai_api_use_v4_for_v3), *([gr.update(visible = '4' not in mod)]*v3_only_len), *([gr.update(visible = '4' in mod)]*v4_only_len) ], inputs = [model], outputs=[vibes_v4,*self.v3_only_ui, *self.v4_only_ui] )
       
        
        for i in range(self.vibe_count_v4):
            # vid,reference_information_extracted,reference_strength,venable
            self.infotext_fields.append((vibe_fields4[0][i],f'{PREFIX} Vibe ID {i+1}'))
            self.infotext_fields.append((vibe_fields4[1][i],f'{PREFIX} Vibe IE {i+1}'))
            self.infotext_fields.append((vibe_fields4[2][i],f'{PREFIX} Vibe Strength {i+1}'))
            self.infotext_fields.append((vibe_fields4[3][i],f'{PREFIX} Vibe On {i+1}'))
    
        vibe_fields4 = [f for field in vibe_fields4 for f in field]
        self.vibe_field_count_v4 = len(vibe_fields4)
        return vibe_fields4
        
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
        self.crop = None
        self.init_images = None
        self.has_protected_prompt = False
        self.incurs_cost = False
        
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
    
    def limit_costs(self, p, cost_limiter = True, nai_batch = False, arbitrary_res = False):
        MAXSIZE = 1024*1024
        if p.width * p.height > MAXSIZE:
            if not cost_limiter: 
                self.incurs_cost = True
                return
            scale = p.width/p.height
            p.height= int(math.sqrt(MAXSIZE/scale))
            p.width= int(p.height * scale)
            if not arbitrary_res:
                p.width = int(p.width/64)*64
                p.height = int(p.height/64)*64
                if not self.mask:
                    if p.width/p.height>scale and (p.height+64)*p.width <= MAXSIZE: p.height += 64
                    elif (p.width+64)*p.height <= MAXSIZE: p.width += 64
                    elif (p.height+64)*p.width <= MAXSIZE: p.height += 64
            self.comment(p,f"Cost Limiter: Reduce dimensions to {p.width} x {p.height}")
        if nai_batch and p.batch_size > 1:
            if not cost_limiter: 
                self.incurs_cost = True
                return

            p.n_iter *= p.batch_size
            p.batch_size = 1
            self.comment(p,f" Cost Limiter: Disable Batching")
        if p.steps > 28: 
            if not cost_limiter: 
                self.incurs_cost = True
                return
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
        
    
    def vibe_v4_ui(self,model,vibe_fields):    
        
        def AddVibes(model,vibes,ids,enabled):
            message = None
            for k,v in list(vibes.items()): vibes[k] = self.update_vibe_file(v)
            slots=[]                
            for i in range(self.vibe_count_v4):
                id = ids[i]
                if not id: slots.append(i)
                # elif id in vibes: vibes.pop(id)
                
            if len(vibes) > len(slots):
                id = enabled[i]
                if not id: slots.append(i)
                
            if len(vibes) > len(slots): 
                message = f"Error: Loaded {len(slots)} of {len(vibes)}, images slots full"
                print ("Not Enough Empty Slots")
            slots.sort(reverse=True)
            for k,v in vibes.items():
                if not slots: break
                i = slots.pop()
                ids[i] = '*'+k # * Prefix indicates preset str/ie values should be loaded
                enabled[i] = True
            return message
                    
        def CreateVibe4(model, source_image, vibe_name, *args):
            args = list(args)                
            ids = args[:self.vibe_count_v4]
            enabled = args[self.vibe_count_v4:]
            
            if not source_image: 
                print("No Source Image!")
                return "Error: No Source Image", args
            
            vibe = None
            
            if 'naivibeid' in source_image.info: vibe = self.load_vibe_file_by_id(source_image.info['naivibeid'])
            if not vibe: 
                if source_image.mode == 'RGBA': source_image = source_image.convert('RGB')
                vibe = nai_api.create_vibe_file(source_image,vibe_name)
            message = AddVibes(model,{vibe['id']:vibe},ids,enabled)                    
            return [message or gr.update(), *ids, *enabled]
            
        def LoadVibe4(model,vibe_file, source_image, vibe_name, *args):
            # vibes = []
            vibes = {}
            def add_json(f):
                try:
                    js = json.load(BytesIO(f))
                    if not js: return False
                    type = js.get("identifier",None)
                    if type == "novelai-vibe-transfer": vibes[js['id']]=js
                    elif type == "novelai-vibe-transfer-bundle":
                        for v in js.get("vibes",[]): vibes[v['id']]= v
                    return True                        
                except ValueError as e:
                    # print("ValueError reading VibeFile JS",e)
                    pass
                except Exception as e:
                    print("Error reading VibeFile JS",e)

            def add_image(img):
                if 'naidata' not in img.info or not add_json(base64.b64decode(img.info['naidata'].encode())):
                    vf = self.load_vibe_file_by_id(img.info.get('naivibeid', None) or nai_api.create_vibe_file(img)['id'])
                    if vf: 
                        vibes[vf['id']] = vf
                        return
                #TODO: Load Vibe from NAI/SDWEBUI Metadata

            if vibe_file:
                for f in vibe_file: 
                    if add_json(f): continue
                    try: 
                        add_image(Image.open(BytesIO(f)))
                    except Exception as e:
                        print("Error reading VibeFile as Image",e)
                        
            if source_image: add_image(source_image)
            
            if vibe_name:
                for vibe in self.find_vibe_files(vibe_name):
                    vibes[vibe['id']] = vibe

            args = list(args)
            
            ids = args[:self.vibe_count_v4]
            enabled = args[self.vibe_count_v4:]
            
            message = AddVibes(model,vibes,ids,enabled) if vibes else "Error: No Vibes found!"
            
            return [None, message or gr.update(), *ids, *enabled]            
            
        self.vibe_cmds = ["Download...", 'Individual Encodings', 'Bundled Encodings', 'Embed in Images', 'Embed Bundle in Source Image','Delete Vibe with Name/ID From Cache', 'Copy Settings from V3 Fields', 'Try Match Encoding Only Vibes to Source']

        def vibe_v4_dl(model, mode, embed_image, vibe_name, *args):
            import tempfile
            args = list(args)
            ids, ies, sts, ens = [args[i*self.vibe_count_v4:(i+1)*self.vibe_count_v4] for i in range(4)]
            
            message = self.vibe_cmds[0]
            fields = [gr.update()]*self.vibe_count_v4*4
            
            if mode == 5:
                if not vibe_name: return message, gr.update(), *fields
                v = self.find_vibe_file(vibe_name)
                if not v: return "Vibe File Not Found", gr.update(), *fields
                id = v['id']
                
                for i in range(len(ids)):
                    if ids[i] == id: ids[i] = ""
                    
                try:
                    path = os.path.join(self.vibe_preview_dir(), f"{self.preview_file_name(vibe_name)} .{id}.png")
                    if self.vibe_preview_dir() and os.path.exists(path): os.remove(path)
                    path = os.path.join(self.vibe_dir(), f"{id}.png")
                    if os.path.exists(path): os.remove(path)
                    path = os.path.join(self.vibe_dir(), f"{id}.naiv4vibe")
                    if os.path.exists(path): os.remove(path)
                except Exception as e:
                    errors.display(e, "Error Deleting Vibe", full_traceback= True)
                    return f"Error Deleting {vibe_name}", gr.update(), *ids, *ies, *sts, *ens

                return f"Deleted {vibe_name}", gr.update(), *ids, *ies, *sts, *ens                
            
            if mode == 6:
                v3images, v3ies, v3strs = [args[self.vibe_count_v4*4 + self.vibe_count*i:self.vibe_count_v4*4 + self.vibe_count*(i+1)] for i in range(3)]
                for i, img in enumerate(v3images):
                    if not img or i>=len(ids): continue
                    image_byte_array = BytesIO()
                    img.save(image_byte_array, format='PNG')
                    ids[i] = "data:image/png;base64," + base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
                    ies[i] = v3ies[i]
                    sts[i] = v3strs[i]
                    ens[i] = True
                    
                return gr.update(value=message), gr.update(value=[], visible = False), *ids, *ies, *sts, *ens
                
            if mode == 7:
                es = [None]*self.vibe_count_v4
                found = 0
                for i,v in enumerate([self.load_vibe_file_by_id(id) for id in ids]):
                    if not v or v.get('type',None) != 'encoding': continue
                    es[i] = list(list(v['encodings'].values())[0].values())[0]['encoding']
                path = self.vibe_dir()
                for f in os.listdir(path):
                    if not f.lower().endswith('.naiv4vibe'): continue
                    file = os.path.join(path, f)
                    if not os.path.isfile(file): continue                
                    try:
                        with open(file,'r') as naiv4vibe: 
                            js = json.loads(naiv4vibe.read())
                            if js.get('type',None) != 'image': continue
                            for enc in [e for m in js.get('encodings', {}).values() for e in m.values()]:
                                e = enc['encoding']
                                if e and e in es:
                                    ids[es.index(e)] = js['id']
                                    ies[es.index(e)] = float(enc.get('params', {}).get('information_extracted',ies[es.index(e)]))
                                    found+=1
                                    break
                    except Exception as e:
                        print("Error Reading naiv4vibe",e)
                return gr.update(value=f"Found {found}"), gr.update(value=[], visible = False), *ids, *ies, *sts, *ens
                
            if mode == 8:
                found = 0
                for i,v in enumerate([self.load_vibe_file_by_id(id) for id in ids]):
                    if not v or not ens[i] or v.get('type',None) != 'image': continue
                    image = v.get('image', None)
                    if not image: continue
                    try:
                        image = Image.open(BytesIO(base64.b64decode(image)))
                        if image.mode == 'RGBA':
                            found+=1
                            image = image.convert('RGB')
                            name = v.get('name', None)
                            
                            if name and '-' in name and not (v['id'].startswith(name.split('-',1)[0]) and v['id'].endswith(name.split('-',1)[1])):
                                name = name + '-RGB'
                            else:
                                name = None
                                
                            vibe = nai_api.create_vibe_file(image,name)                            
                            if not self.load_vibe_file_by_id(vibe['id']):
                                vibe = self.update_vibe_file(vibe)
                            ids[i] = vibe['id']
                    except Exception as e:
                        print("Error Reading naiv4vibe",e)

                return gr.update(value=f"Found {found}"), gr.update(value=[], visible = False), *ids, *ies, *sts, *ens
            
            
            vibes = [self.load_vibe_file_by_id(id) if ens[i] else None for i,id in enumerate(ids)]
            
            for i in range(len(ids)-1,-1,-1):
                if vibes[i]:
                    ies[i] = nai_api.get_closest_ie(vibes[i], ies[i], model, None)
                if not vibes[i] or not ies[i] and vibes[i]['type'] != 'encoding':
                    vibes.pop(i)
                else:
                    vibes[i]['importInfo'] = {
                        "model": model,
                        "information_extracted": ies,
                        "strength":sts
                    }
                    
            
            # vibes = [vibe for vibe in (self.load_vibe_file_by_id(id) for id in ids) if vibe]
            if not vibes and mode in [1,2,3,4]:
                message = "Nothing to download..."
                mode = 0

            paths= []
            files = []
            match mode:
                case 1:
                    files += [(f"{v['name']} -", v) for v in vibes]
                case 2:
                    bundle = nai_api.create_vibe_bundle(vibes)
                    name = f"{vibes[0]['name']} et al-" if vibes else None
                    files.append((name, bundle))
                case 3:
                    for i,v in enumerate(vibes):
                        name = f"{v['name']} -"
                        try: 
                            image = Image.open(BytesIO(base64.b64decode(v['image'].encode())))
                            pnginfo = PngImagePlugin.PngInfo()
                            pnginfo.add_text('naidata', base64.b64encode(json.dumps(v).encode()).decode("utf-8"))
                            t = tempfile.NamedTemporaryFile(delete=False, prefix=name, suffix=".png")
                            image.save(t,format='PNG',pnginfo=pnginfo)
                            paths.append(t.name)                        
                        except Exception: files.append((name,v))                    
                case 4:
                    bundle = nai_api.create_vibe_bundle(vibes)
                    name = f"{vibes[0]['name']} et al-" if vibes else None
                    if not embed_image: files.append((name, bundle))
                    else:
                        pnginfo = PngImagePlugin.PngInfo()
                        pnginfo.add_text('naidata', base64.b64encode(json.dumps(bundle).encode()).decode("utf-8"))
                        t = tempfile.NamedTemporaryFile(delete=False, prefix=name, suffix=".naiv4vibebundle.png")
                        embed_image.convert("RGBA").save(t, format='PNG', pnginfo=pnginfo)
                        paths.append(t.name)
                case _:
                    return message, gr.update(), *fields
                    
            for name, file in files:
                t = tempfile.NamedTemporaryFile(delete=False, prefix=name, suffix=".naiv4vibebundle")
                t.write(json.dumps(file).encode())
                paths.append(t.name)
                
            return message, gr.update(value=paths, visible = True), *fields        

        vibe_fields4,vibes_v4,normalize_reference_strength_multiple,normalize_negatives,normalize_level,source_image,vibe_file,vibe_name,create_vibe,load_vibe,download_dd_vibe,vibe_file_dl, *extra_fields = self.vibe_v4_main_ui(model)

        load_vibe.click(fn= LoadVibe4, inputs=[model,vibe_file, source_image, vibe_name, *vibe_fields4[0] , *vibe_fields4[3] ], outputs=[vibe_file,download_dd_vibe,*vibe_fields4[0], *vibe_fields4[3]])
        
        vibe_file.upload(fn= lambda model,vibe_file,*args:LoadVibe4(model,vibe_file,None, None,*args), inputs=[model,vibe_file, *vibe_fields4[0] , *vibe_fields4[3] ], outputs=[vibe_file,download_dd_vibe,*vibe_fields4[0], *vibe_fields4[3]])        
        
        create_vibe.click(fn= CreateVibe4, inputs=[model, source_image, vibe_name, *vibe_fields4[0] , *vibe_fields4[3] ], outputs=[download_dd_vibe,*vibe_fields4[0], *vibe_fields4[3]])
        
        vb_dl_js = 'function(...args){if ( args[1].toLowerCase().includes("delete") && !window.confirm("Delete Vibe Encodings From Cache? ID/Name: "+args[3]+"")) { args[3] = ""; args[1] = "Cancelled..." } return args; }'
        download_dd_vibe.input(fn= vibe_v4_dl,_js= vb_dl_js,  inputs = [model,download_dd_vibe,source_image, vibe_name,*vibe_fields4[0],*vibe_fields4[1],*vibe_fields4[2],*vibe_fields4[3] ,*vibe_fields[0],*vibe_fields[1],*vibe_fields[2] ], outputs = [download_dd_vibe, vibe_file_dl,*vibe_fields4[0],*vibe_fields4[1],*vibe_fields4[2],*vibe_fields4[3] ])
        
        vibe_file_dl.clear(fn=lambda:gr.update(visible=False) , outputs = [vibe_file_dl], show_progress=False)
        
        return vibe_fields4, vibes_v4, normalize_reference_strength_multiple,normalize_negatives,normalize_level, *extra_fields 
        
    
    def vibe_v4_main_ui(self, model):
        vibe_fields4 = []
        
        def dovibe4(idx):
            fields = self.vibe_v4_entry(model, idx+1)
            for fi in range(len(fields)):
                if fi >= len(vibe_fields4): vibe_fields4.append([])
                vibe_fields4[fi].append(fields[fi])
            if idx+1<self.vibe_count_v4: dovibe4(idx+1)                

        with gr.Accordion(label=f'Vibe Transfer', open=True, elem_id=f"nai_vibe_v4_{elempfx}") as vibes_v4:
            with gr.Row():
                normalize_reference_strength_multiple = gr.Checkbox(value=True, label=f"Normalize",min_width=64)
                normalize_negatives = gr.Dropdown(value="Abs", label=f"Negatives", choices= ["Abs", "Zero", "Ignore"], type="index", visible = True,min_width=64)
                normalize_level = gr.Slider(value = 0, minimum=0.0, maximum = 3.0, step = 0.01, label = "Level", min_width=64)
            with gr.Accordion(label=f'Add/Create/Download Vibes', open=False):
                with gr.Row():                        
                    vibe_file = gr.File(label = "Load Vibe Files",file_count='multiple', type='binary', file_types=['naiv4vibebundle','naiv4vibe','png'],min_width=64)
                    source_image = gr.Image(label=f"Source Image", elem_id=f"vibe_src_image_{elempfx}", show_label=True, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA",min_width=64)
                with gr.Row():
                    vibe_name = gr.Textbox(label = "Name/ID",scale =100,max_lines=1,container=True, placeholder="Name/ID", show_label=False)
                    load_vibe = ToolButton(ui.paste_symbol, elem_id=f"load_vibe_{elempfx}",scale =1, tooltip="Load Existing Vibe from Name/ID or Image")
                    create_vibe = ToolButton('🖼️', elem_id=f"create_vibe_{elempfx}",scale =1, tooltip="Create New Vibe from Image")
                    
                download_dd_vibe = gr.Dropdown(value=self.vibe_cmds[0], choices= self.vibe_cmds ,type="index", label="Mode",show_label=False)
                
                vibe_file_dl = gr.File(label = "Downloads",file_count='multiple', interactive=True,visible = False)
            with gr.Row():
                dovibe4(0)            
            # dovibe4(0)            
            
        return vibe_fields4,vibes_v4,normalize_reference_strength_multiple,normalize_negatives,normalize_level,source_image,vibe_file,vibe_name,create_vibe,load_vibe,download_dd_vibe, vibe_file_dl

    
    def vibe_v4_entry_ui(self, model, idx):    
        self.ie_list_default = "Encoded IE Values..."
    
        with gr.Column(elem_id=f"{idx}_vb4_group_{elempfx}", visible = False) as group:
            with gr.Row(variant="compact"):
                venable = gr.Checkbox(value=True, label=f"V{idx}",show_label=False,scale = 1,min_width=16)
                vname =gr.Textbox(label=f"V{idx} Name:", visible = True, value = "",container=False,scale = 80,max_lines=1)
                rename = ToolButton(ui.save_style_symbol,scale = 1,visible=False)                        
                remove = ToolButton(ui.clear_prompt_symbol,scale = 1)
                
            vid =gr.Textbox(label=f"V{idx} ID:", visible = False, value = "",container=False)
            with gr.Row():
                with gr.Column(scale= 2):
                    reference_strength=gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, label=f'Reference Strength', elem_id=f"v4_str_{idx}_{elempfx}", value=0.6,min_width=128)
                    reference_information_extracted = gr.Slider(minimum=0.0, maximum=1.0, step= 0.01, label=f'Information Extracted', elem_id=f"v4_ie_{idx}_{elempfx}", value=1.0,min_width=128)
                        
                    with gr.Row() as encode_row:
                        encoding_values = gr.Dropdown(value=self.ie_list_default, choices= [self.ie_list_default],type="value", label="Encodings",show_label=False,min_width=128)
                        
                        encodebutton = gr.Button("Generate: -2 Anals", visible = False,min_width=128)
                with gr.Column(min_width=64):
                    preview_image = gr.Image(label=f"Preview Image {idx}", elem_id=f"preview_image_{idx}_{elempfx}", show_label=False,visible = True,  interactive=False, type="pil", tool="editor", image_mode="RGBA",scale= 1)
            self.v4_only_ui.append(encode_row)
        return group,vid,reference_information_extracted,reference_strength,venable,vname,preview_image,rename,remove,encodebutton,encoding_values
    
    def get_vibe_ie_entries(self, vibe, model= ""):
        choices = [self.ie_list_default]
        if vibe is None: return choices

        mk = nai_api.vibe_model_names.get(model, None)
        if mk and mk in vibe.get('encodings', {}):                        
            if mk is not None and len(list(vibe.get('encodings', {}).keys())) > 1: choices.append(mk)
            for e in vibe.get('encodings', {}).get(mk,{}).values():
                ie = e.get('params', {}).get('information_extracted',None)
                if ie is not None: choices.append(str(ie))
        for m,es in vibe.get('encodings', {}).items():
            if m == mk: continue
            choices.append(m)
            for e in es.values():
                ie = e.get('params', {}).get('information_extracted',None)
                if ie is not None: choices.append(str(ie))
        return choices
        
    def vibe_v4_entry(self, model, idx):
        group,vid,reference_information_extracted,reference_strength,venable,vname,preview_image,rename,remove,encodebutton,encoding_values,*extra_fields = self.vibe_v4_entry_ui(model, idx)
        
        def RenameVibe4(vid,vname, model, reference_information_extracted,reference_strength):
            self.rename_vibe_file(vid, vname, model, reference_information_extracted, reference_strength)
            return gr.update(visible = False)
            
        def GenerateEncoding(model,vid,ie):
            vibe = self.find_vibe_file(vid)
            choices = self.get_vibe_ie_entries(None)
            if not vibe: 
                print("Vibe File Not Found")
                return gr.update(), gr.update(choices = choices, value = choices[0])
            result, vibe = self.vibe_request_encoding(vibe, ie, model)
            
            if result:
                return gr.update(visible = False), gr.update(choices = self.get_vibe_ie_entries(vibe, model), value = choices[0])
            return gr.update(), gr.update()
            
        def UpdateVibe4(model,vid,reference_information_extracted,vname,preview_image,*args):
            if not vid: return gr.update(), gr.update(visible=False), "", gr.update(value=None), gr.update(interactive=True), gr.update(), gr.update(visible = False), gr.update()
            is_new=False
            
            ieval= None
            strval = None
            
            if vid.startswith('*'):
                is_new=True
                vid = vid[1:]
                reference_information_extracted = 1.0

            if vid.startswith('data:image/'):
                txt = vid.split(',',1)[1]
                img = Image.open(BytesIO(base64.b64decode(txt)))
                vibe = self.load_vibe_file_by_id(getattr(img,'info',{}).get('naivibeid', None) or nai_api.create_vibe_file(img)['id'])
                if not vibe: 
                    vibe = self.update_vibe_file(nai_api.create_vibe_file(img))
                vid = vibe['id']
                ieval = float(reference_information_extracted)
            elif len(vid)> 2560: #ludicrously long id is probably a whole encoding. 
                vibe = self.vibe_create_encoding(vid,model)
                if vibe: vid = vibe['id']
            else:            
                vibe = self.find_vibe_file(vid)

            if not vibe and len(vid)> 2560:
                vname = f"Unrecognized Encoding"
                preview_image = None
                ie = gr.update(interactive=True)
                is_encoding = True                
            elif not vibe: 
                print("Vibe File Not Found")
                vname = f"Unknown ID: {vid}"
                preview_image = None
                ie = gr.update(interactive=True)
                is_encoding = False
            else:
                is_encoding = vibe['type'] == 'encoding'
                preview_image = nai_api.get_vibe_preview(vibe)
                vname=vibe['name']
                if is_new: ieval, strval = nai_api.get_vibe_presets(vibe,model,None)
                if '4' in model: ieval = nai_api.get_closest_ie(vibe, reference_information_extracted, model, None)
                ie = gr.update(interactive= not is_encoding, value = ieval if '4' in model and ieval is not None else reference_information_extracted)
            choices = self.get_vibe_ie_entries(vibe, model)
            return vid, gr.update(visible=True), vname,gr.update(value=preview_image), ie, gr.update() if strval is None else strval,gr.update(visible = not is_encoding and ieval is None), gr.update(visible = not is_encoding, choices = choices, value = choices[0])
                
        def UpdateVibeIE4(model,vid,reference_information_extracted,*args):
            # if len(vid)> 2560: return gr.update(), gr.update(visible=False)
            vibe = self.load_vibe_file_by_id(vid)
            if not vibe: 
                return gr.update(), gr.update()
            if vibe['type'] == 'encoding':
                return gr.update(value = nai_api.get_vibe_ie(vibe,model), interactive = False), gr.update(visible=False)
            has_encoding = nai_api.has_encoding(vibe, reference_information_extracted, model)
            return gr.update(interactive = True), gr.update(visible = not has_encoding)

            
        vid.change(fn = UpdateVibe4, inputs = [model,vid, reference_information_extracted,vname,preview_image], outputs = [vid,group, vname,preview_image,reference_information_extracted,reference_strength,encodebutton,encoding_values])
        vname.input(lambda: gr.update(visible = True), outputs = [rename])
        
        venable.change(fn=None , _js =f'function(ven){{gradioApp().getElementById("{f"{idx}_vb4_group_{elempfx}"}").style.opacity = ven ? 1.0 : 0.45;}}' , inputs=[venable], show_progress=False)
        
        remove.click(fn=lambda:"", outputs=[vid])
        rename.click(fn=RenameVibe4, inputs = [vid, vname, model, reference_information_extracted,reference_strength], outputs=[rename])
        vname.submit(fn=RenameVibe4, inputs = [vid, vname, model, reference_information_extracted,reference_strength], outputs=[rename])
        
        encodebutton.click(fn=GenerateEncoding, inputs = [model, vid,reference_information_extracted],outputs=[encodebutton,encoding_values])
        reference_information_extracted.change(fn = UpdateVibeIE4, inputs = [model,vid,reference_information_extracted], outputs = [reference_information_extracted,encodebutton], show_progress=False)
        reference_information_extracted.release(fn = UpdateVibeIE4, inputs = [model,vid,reference_information_extracted], outputs = [reference_information_extracted,encodebutton], show_progress=False)                    
        
        encoding_values.input(fn=lambda value: [self.get_vibe_ie_entries(None)[0], nai_api.tryfloat(value,  gr.update())], inputs=[encoding_values], outputs=[encoding_values,reference_information_extracted])
                
        return vid,reference_information_extracted,reference_strength,venable,*extra_fields

    def vibe_dir(self, encoding_only = False):    
        path = shared.opts.nai_api_vibe_v4_directory 
        if encoding_only and shared.opts.nai_api_vibe_v4_encoding_directory and os.path.exists(shared.opts.nai_api_vibe_v4_encoding_directory):
            path = shared.opts.nai_api_vibe_v4_encoding_directory
        if not path or not os.path.exists(path):
            path = os.path.join(base_dir,'vibe_encodings')
            if not os.path.exists(path): os.mkdir(path)
        return path

    def vibe_preview_dir(self):
        path = shared.opts.nai_api_vibe_v4_preview_directory
        if not path or not os.path.exists(path):
            path = os.path.join(base_dir,'vibe_images')
            if not os.path.exists(path): os.mkdir(path)
        return path
        
    def vibe_create_encoding(self, encoding, model):
        vibe = nai_api.create_encoding_file(encoding,model)
        if not vibe: return vibe
        id = vibe['id']
        path=os.path.join(self.vibe_dir(True),id + '.naiv4vibe')
        if not os.path.exists(path):
            with open(path,'w') as file:
                file.write(json.dumps(vibe,separators=(',\n', ': ')))
        return vibe
        
    def vibe_request_encoding(self, vibe, ie, model):
        if nai_api.has_encoding(vibe, ie, model):
            return True, vibe
        if nai_api.add_encoding(self.get_api_key(), vibe, ie, model):
            return True, self.update_vibe_file(vibe)
        return False, vibe

    def preview_file_name(self, name):
        return re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F\.]", "_", name).strip()
        
    def save_vibe_images(self,dest):
        type = dest["type"]
        name = dest.get("name","")
        id = dest.get("id","")
        if 'image' in dest and type != 'encoding' and name:
            format = Image.registered_extensions()['.png']
            path = os.path.join(self.vibe_preview_dir(), f"{self.preview_file_name(name)} .{id}.png")
            if self.vibe_preview_dir() and not os.path.exists(path):
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text('naivibeid', id)
                prev = Image.open(BytesIO(base64.b64decode(dest['image'].encode())))
                prev.save(path,format=format,pnginfo=pnginfo)                    
            path = os.path.join(self.vibe_dir(), f"{id}.png")
            if not os.path.exists(path) and dest.get('thumbnail', ""):
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text('naivibeid', id)
                prev = Image.open(BytesIO(base64.b64decode(dest['thumbnail'].split(',',1)[-1].encode())))
                prev.save(path,format=format,pnginfo=pnginfo)
        
    def update_vibe_file(self, vibe):
        id = vibe.get("id","")
        if not id: 
            print ("No ID!")
            return vibe
        is_encoding = vibe["type"] == 'encoding'
        dest = self.load_vibe_file_by_id(id)        
        if not dest:
            dest = vibe.copy()
            if 'importInfo' in dest: del dest['importInfo']
            if not dest.get("name",""): dest['name'] = f"{id[:6]}-{id[-6:]}"
            if not is_encoding: self.save_vibe_images(dest)
        elif is_encoding: 
            return dest
        else:
            de = dest.get('encodings')
            for mk, md in vibe.get('encodings', {}).items():
                if mk not in de: de[mk]=md.copy()
                else: de[mk].update(md)

        with open(os.path.join(self.vibe_dir(is_encoding),id + '.naiv4vibe'),'w') as file:
            file.write(json.dumps(dest,separators=(',\n', ': ')))
        return dest
        
    def rename_vibe_file(self, id, newname, model = None, ie = None, st= None):
        if not newname: newname = f"{id[:6]}-{id[-6:]}"
        dest = self.load_vibe_file_by_id(id)
        if not dest:
            print("Could Not Rename")
            return 
        name = dest.get("name","")
        is_encoding = dest["type"] == 'encoding'
        path = os.path.join(self.vibe_dir(is_encoding),id + '.naiv4vibe')
        dest['name'] = newname
        
        if model and ie is not None and st is not None:
            dest['importInfo']={
                "model": model,
                "information_extracted": ie,
                "strength": st
            }
        elif name == newname: return
        
        with open(path,'w') as file:
            file.write(json.dumps(dest,separators=(',\n', ': ')))            
        if is_encoding or self.preview_file_name(name).lower() == self.preview_file_name(newname).lower(): return        
        try:
            path = os.path.join(self.vibe_preview_dir(), f"{self.preview_file_name(name)} .{id}.png")
            if self.vibe_preview_dir() and os.path.exists(path): os.remove(path)
            path = os.path.join(self.vibe_dir(), f"{id}.png")
            if os.path.exists(path): os.remove(path)
        except Exception as e:
            print (e)
        self.save_vibe_images(dest)        
        return
        
    def load_vibe_file_by_id(self, id):
        if not id: return None
        return self.load_vibe_file(self.find_vibe_path_by_id(id))

    def load_vibe_file(self, path):
        if not path or not os.path.exists(path): return None
        with open(path,'r') as file:
            return json.loads(file.read())
    
    def find_vibe_path_by_id(self, id):
        id = id.strip()
        path = os.path.join(self.vibe_dir(), id + '.naiv4vibe')
        if os.path.exists(path): return path
        path2 = os.path.join(self.vibe_dir(True), id + '.naiv4vibe')
        if path != path2 and os.path.exists(path2): return path2
        return None        

    def find_vibe_files(self, name):
        name = name.strip()
        vibe = self.load_vibe_file_by_id(name)
        if vibe: return [vibe]
        filename = self.preview_file_name(name).lower()        
        vibes = []        
        def check_path(path):
            for f in os.listdir(path):
                if not f.lower().endswith('.naiv4vibe'): continue
                file = os.path.join(path, f)
                if not os.path.isfile(file): continue                
                try:
                    with open(file,'r') as naiv4vibe: 
                        js = json.loads(naiv4vibe.read())
                        if self.preview_file_name(js.get('name',"")).lower()==filename:
                            vibes.append(js)
                except Exception as e:
                    print("Error Reading naiv4vibe",e)
                    
        check_path(self.vibe_dir())
        if self.vibe_dir()!=self.vibe_dir(True):
            check_path(self.vibe_dir(True))
        return vibes

    def find_vibe_file(self, name):
        name = name.strip()
        vibe = self.load_vibe_file_by_id(name)
        if vibe: return vibe
        filename = self.preview_file_name(name).lower()
        
        if self.vibe_preview_dir():
            for f in os.listdir(self.vibe_preview_dir()):
                if not f.lower().endswith('.png'): continue
                sp = f.split('.',2)
                if sp[0].strip().lower()==filename:
                    id = sp[1].strip().lower()
                    vibe = self.load_vibe_file_by_id(id)
                    if vibe: return vibe
        
        def check_path(path):
            for f in os.listdir(path):
                if not f.lower().endswith('.naiv4vibe'): continue
                file = os.path.join(path, f)
                if not os.path.isfile(file): continue                
                with open(file,'r') as naiv4vibe: 
                    try:
                        js = json.loads(naiv4vibe.read())
                        if self.preview_file_name(js.get('name',"")).lower()==filename:
                            return js
                    except Exception as e:
                        print("Error Reading naiv4vibe",e)
            return None
            
        vibe = check_path(self.vibe_dir())
        if not vibe and self.vibe_dir()!=self.vibe_dir(True):
            check_path(self.vibe_dir(True))
        return vibe
            
    def nai_configuration(self,p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,reclrLvlAlpha,deliberate_euler_ancestral_bug,prefer_brownian,legacy_uc,normalize_reference_strength_multiple,normalize_negatives,normalize_level,cref_image,cref_style,cref_fidel,keep_mask_for_local,*args):
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
            if self.augment_mode == 'bg-removal':
                if cost_limiter:
                    self.comment(p,f"Cost Limiter: Background Removal is never free, switching to Declutter")
                    self.augment_mode='declutter'       
                else: self.incurs_cost = True
            self.limit_costs(p, cost_limiter, arbitrary_res = True)
                
        else: 
            self.augment_mode = ""
            self.limit_costs(p, cost_limiter)
            self.adjust_resolution(p)
            self.setup_sampler_name(p, sampler,getattr(p,f'{PREFIX}_'+ 'noise_schedule',noise_schedule))
            
            
        if cref_image:        
            self.cref_image = cref_image
            self.cref_style = cref_style
            self.cref_fidel = cref_fidel
            if cost_limiter:
                self.comment(p,f"Cost Limiter: Disabling Character Reference.")
                self.cref_image=None       
            # else: self.incurs_cost = True
        else:
            self.cref_image = None
            self.cref_style = None
            self.cref_fidel = 0



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
        self.isV4 = '4' in self.model
        
        if do_local_img2img == 1 or do_local_img2img == 2:
            self.set_local(p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,reclrLvlAlpha,deliberate_euler_ancestral_bug,prefer_brownian,legacy_uc,normalize_reference_strength_multiple,normalize_negatives,normalize_level,cref_image,cref_style,cref_fidel,keep_mask_for_local)
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
        
    def set_local(self,p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,reclrLvlAlpha,deliberate_euler_ancestral_bug,prefer_brownian,legacy_uc,normalize_reference_strength_multiple,normalize_negatives,normalize_level,cref_image,cref_style,cref_fidel,keep_mask_for_local,*args):    
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

    def nai_preprocess(self,p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,reclrLvlAlpha,deliberate_euler_ancestral_bug,prefer_brownian,legacy_uc,normalize_reference_strength_multiple,normalize_negatives,normalize_level,cref_image,cref_style,cref_fidel,keep_mask_for_local,*args):
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
                if not p.inpaint_full_res: 
                    mask = images.resize_image(p.resize_mode if p.resize_mode < 3 else 0, mask, p.width, p.height)
                elif p.init_images and p.init_images[0].size != mask.size:
                    mask = mask.resize(p.init_images[0].size,Image.Resampling.NEAREST)

                if not alt_masking:
                    if self.inpaint_mode == 1 or not self.isV4:
                        if p.mask_blur > 0: mask = maskblur(mask, p.mask_blur)
                        if p.inpaint_full_res:
                            overlay_mask = mask
                            crop = masking.expand_crop_region(masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding), p.width, p.height, mask.width, mask.height)
                            mask = images.resize_image(2, mask.crop(crop), p.width, p.height)
                        else:
                            mask = images.resize_image(p.resize_mode if p.resize_mode < 3 else 0, mask, p.width, p.height)
                            overlay_mask = Image.fromarray(np.clip((np.array(mask).astype(np.float32)) * 2, 0, 255).astype(np.uint8))
                    else:
                        if p.mask_blur > 0: mask = maskblur(mask, p.mask_blur)
                        dilate = 4
                        radius = 20
                        overlay_mask = mask.resize((mask.width//8, mask.height//8),Image.Resampling.NEAREST)
                        overlay_mask = clip(overlay_mask, 128)
                        if dilate > 0: overlay_mask = dilate_contract(overlay_mask, dilate)
                        overlay_mask = overlay_mask.resize((mask.width, mask.height),Image.Resampling.NEAREST)
                        if dilate > 0: overlay_mask = maskblur2(overlay_mask, radius,2)
                        else: overlay_mask = blur_out(overlay_mask, radius)
                        
                        if p.inpaint_full_res:
                            crop = masking.expand_crop_region(masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding), p.width, p.height, mask.width, mask.height)
                            mask = images.resize_image(2, mask.crop(crop), p.width, p.height).convert('L')

                        mask = mask.resize((p.width//8, p.height//8),Image.Resampling.NEAREST)
                        mask = mask.resize((p.width, p.height),Image.Resampling.NEAREST)
                        mask = clip(mask, 128)
                        
                    min_mask = mask                    
                else: 
                    adjust_mask = getattr(p,f'{PREFIX}_'+ 'adjust_mask',0)                    
                    blur_outside_mask = getattr(p,f'{PREFIX}_'+ 'blur_outside_mask',False)
                    if p.inpaint_full_res:
                        crop = masking.expand_crop_region(masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding + p.mask_blur*2+expand*2), p.width, p.height, mask.width, mask.height)
                        mask = images.resize_image(2, mask.crop(crop), p.width, p.height)
                    min_mask = mask.convert('L')
                    
                    if adjust_mask: mask = dilate_contract(mask,adjust_mask)
                    if adjust_mask < 0: min_mask = mask.convert('L')
                    # if nai_mask_fill_blur > 8: min_mask = blur_out(min_mask, min(adjust_mask-8, adjust_mask//4))
                    mask_fill_blur = getattr(p,f'{PREFIX}_'+ 'mask_fill_blur',0)
                    if mask_fill_blur > 0:
                        min_mask = blur_out(min_mask, min(mask_fill_blur, adjust_mask))
                    
                    if p.mask_blur > 0: 
                        overlay_mask = maskblur(mask, p.mask_blur) if not blur_outside_mask else blur_out(mask, p.mask_blur)
                    else: overlay_mask = mask
                    
                    if self.isv4 and self.inpaint_mode != 1:
                        mask = mask.resize((mask.width//8, mask.height//8),Image.Resampling.NEAREST)
                        mask = mask.resize((mask.width*8, mask.height*8),Image.Resampling.NEAREST)
                        mask = clip(mask, 128)

                mask = mask.convert('L')
                self.mask_for_overlay = overlay_mask
                self.overlay_mask = ImageOps.invert(overlay_mask.convert('L'))

            for i in range(len(p.init_images)):
                image = p.init_images[i]
                if shared.opts.data.get('save_init_img',False) and not hasattr(p,'enable_hr'):
                    self.save_init_img(image)
                image = images.flatten(image, shared.opts.img2img_background_color)
                
                if not crop and (image.width != p.width or image.height != p.height):
                    image = images.resize_image(p.resize_mode if p.resize_mode < 3 else 0, image, p.width, p.height)
                
                if crop: image = images.resize_image(2, image.crop(crop), p.width, p.height)
 
                if mask and p.inpainting_fill != 1 and not self.augment_mode:
                    image.paste(masking.fill(image, min_mask).convert("RGBA").convert("RGBa"), mask=min_mask.convert('L'))

                init_images.append(image)
                
            self.mask = mask
            self.crop = crop
            self.init_images = init_images
        else: 
            self.init_images = None
            self.mask = None
            self.crop = None
            
        if self.augment_mode == 'recolorize':
            for i in range(len(self.init_images)):
                g = nai_api.GrayLevels(self.init_images[i],reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut)
                if reclrLvlAlpha < 1:
                    g = Image.blend(self.init_images[i],b.convert("RGB"),reclrLvlAlpha)
                self.init_images[i] = g
        
        # TODO: Skip this when augment is enabled.
        self.reference_image = []
        self.reference_information_extracted = []
        self.reference_strength = []        
        self.use_v4_vibe = self.isV4 or shared.opts.nai_api_use_v4_for_v3
        self.normalize_reference_strength_multiple = self.use_v4_vibe and normalize_reference_strength_multiple

        if self.use_v4_vibe:
            vids, ies, sts, ens = [args[-(self.vibe_field_count+self.vibe_field_count_v4):-self.vibe_field_count][i*self.vibe_count_v4:(i+1)*self.vibe_count_v4] for i in range(self.vibe_field_count_v4//self.vibe_count_v4)]
            
            for i in range(1,1+self.vibe_count_v4):
                id = getattr(p, f'{PREFIX}_VibeID_{i}', vids[i-1])
                en = getattr(p, f'{PREFIX}_VibeOn_{i}', ens[i-1])
                ie = getattr(p, f'{PREFIX}_VibeExtract_{i}', ies[i-1])
                st = getattr(p, f'{PREFIX}_VibeStrength_{i}', sts[i-1])
                ie = nai_api.tryfloat(ie,1.0)
                st = nai_api.tryfloat(st,0.6)
                if not id: continue
                vibe = self.load_vibe_file_by_id(id)
                if vibe is None:
                    self.comment(p,f"Could not find file for Vibe{i} ID: {id}")
                    en=False
                elif not self.isV4 and en:
                    if not vibe.get('image', None):
                        self.comment(p,f"Skipping Encoding Only Vibe{i}, no image for for v3")
                        en = False
                    else: 
                        self.reference_image.append(vibe['image'])
                elif en:
                    if vibe.get('type',"") == 'encoding':
                        enc, ie = nai_api.get_closest_encoding(vibe, ie, self.model)
                        if not enc:
                            self.comment(p,f"Skipping Vibe{i}, encoding does not match selected model")
                            en = False
                    else:
                        enc = nai_api.get_encoding(vibe, ie, self.model)
                        if not enc and (not self.cost_limiter or shared.opts.nai_api_vibe_always_encode):
                            self.incurs_cost = True
                            rslt, vibe = self.vibe_request_encoding(vibe, ie, model)
                            if rslt:
                                self.comment(p,f"Spent 2 anals encoding Vibe{i} at {ie}")
                                enc = nai_api.get_encoding(vibe, ie, self.model)
                            else:
                                self.comment(p,f"Could not generate Encoding for Vibe{i} ID: {id}")
                        if not enc:
                            enc, newie = nai_api.get_closest_encoding(vibe, ie, self.model)
                            if enc:
                                self.comment(p,f"No encoding for Vibe{i} at IE:{ie}, using IE:{newie}")
                                ie = newie
                    if not enc:
                        self.comment(p,f"Skipping Vibe{i}, encoding does not exist")
                        en = False
                    else:
                        self.reference_image.append(enc)
                    
                if en:
                    self.reference_information_extracted.append(ie)
                    self.reference_strength.append(st)

                p.extra_generation_params[f'{PREFIX} Vibe On {i}'] = en
                p.extra_generation_params[f'{PREFIX} Vibe ID {i}'] = id
                p.extra_generation_params[f'{PREFIX} Vibe IE {i}'] = ie
                p.extra_generation_params[f'{PREFIX} Vibe Strength {i}'] = st
                
            if self.isV4 and self.cost_limiter and len(self.reference_image) > 4:
                self.reference_image = self.reference_image[:4]
                self.reference_information_extracted = self.reference_information_extracted[:4]
                self.reference_strength = self.reference_strength[:4]
                self.comment(p,f"Cost Limiter: Reducing number of Vibe images to 4")
                
            p.extra_generation_params[f'{PREFIX} '+ 'normalize_reference_strength'] = normalize_reference_strength_multiple          
            
            if normalize_reference_strength_multiple and (normalize_negatives in [1,2] or normalize_level>0):
                total = sum( max(0,n) if normalize_negatives in [1,2] else abs(n) for n in self.reference_strength)
                if normalize_level and total > 0 or total > 1.0:
                    if normalize_level: total /= normalize_level
                    for i in range(len(self.reference_image)):
                        if self.reference_strength[i] > 0 or normalize_negatives == 1: self.reference_strength[i] /= total                        
                p.extra_generation_params[f'{PREFIX} '+ 'normalize_negatives'] = {1:"Zero",2:"Ignore"}.get(normalize_negatives,"Abs")
                p.extra_generation_params[f'{PREFIX} '+ 'normalize_level'] = normalize_level
                self.normalize_reference_strength_multiple = False
                
        if not self.use_v4_vibe or not self.isV4 and not self.reference_image:
            self.use_v4_vibe = False
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
            
    def convert_to_nai(self, prompt, neg,convert_prompts="Always" , use_numeric_emphasis=False):
        if convert_prompts != "Never":
            if convert_prompts == "Always" or nai_api.prompt_has_weight(prompt): 
                prompt = nai_api.sd_to_nai_v4(prompt) if use_numeric_emphasis else nai_api.prompt_to_nai(prompt,True) 
            if convert_prompts == "Always" or nai_api.prompt_has_weight(neg): 
                neg = nai_api.sd_to_nai_v4(neg) if use_numeric_emphasis else nai_api.prompt_to_nai(neg,True)

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
            return prompt if index > 0 or text_tag else original_prompt , text_tag
            
        prompt, text_tag = parse(prompt,False)
        neg, _ = parse(neg,True)
        
        return prompt, neg, chars, text_tag
            
    def infotext(self,p,i):
        iteration = int(i / p.batch_size)
        batch = i % p.batch_size
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, iteration, batch)


    def nai_generate_images(self,p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,reclrLvlAlpha,deliberate_euler_ancestral_bug,prefer_brownian,legacy_uc,normalize_reference_strength_multiple,normalize_negatives,normalize_level,cref_image,cref_style,cref_fidel,keep_mask_for_local,*args):
        if self.disabled or p.nai_processed is not None: return 
        DEBUG_LOG("nai_generate_images")
        DEBUG_LOG(enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,reclrLvlAlpha,deliberate_euler_ancestral_bug,prefer_brownian,legacy_uc,normalize_reference_strength_multiple,normalize_negatives,normalize_level,cref_image,cref_style,cref_fidel,keep_mask_for_local)
        
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
        extra_noise = max(getattr(p,"extra_noise",0) , extra_noise)
        
        defry = getattr(p,f'{PREFIX}_'+ 'defry',defry)        
        emotion = getattr(p,f'{PREFIX}_'+ 'emotion',emotion)
        inpaint_mode=self.inpaint_mode
        
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
            
            p.extra_generation_params[f'{PREFIX} '+ 'legacy_uc'] = legacy_uc            
            
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
            
            image= None if ( not isimg2img or self.do_local_img2img == 1 or self.init_images is None or len(self.init_images) == 0) else self.init_images[i % len(self.init_images)]
                
            if self.isV4: 
                prompt,neg,chars,text_tag = self.parse_tags(p.all_prompts[i],  p.all_negative_prompts[i])
                for c in chars:
                    cp,cn = self.convert_to_nai(c.get('prompt',''),c.get('uc', ''),convert_prompts,shared.opts.nai_api_use_numeric_emphasis)
                    if cp: c['prompt'] = cp
                    if cn: c['uc'] = cn
                prompt,neg = self.convert_to_nai(prompt,neg,convert_prompts,shared.opts.nai_api_use_numeric_emphasis)
            else: 
                prompt,neg = self.convert_to_nai(p.all_prompts[i],  p.all_negative_prompts[i], convert_prompts, False)
                chars = []
                text_tag = None
            
            if self.augment_mode:
                return nai_api.AugmentParams('colorize' if self.augment_mode == 'recolorize' else self.augment_mode,image,p.width,p.height,prompt,defry,emotion,seed)
                
            return nai_api.NAIGenParams(prompt, neg, seed=seed , width=p.width, height=p.height, scale=p.cfg_scale, sampler = self.sampler_name, steps=p.steps, noise_schedule=self.noise_schedule,sm= "smea" in str(smea).lower(), sm_dyn="dyn" in str(smea).lower(), cfg_rescale=cfg_rescale,uncond_scale=0 ,dynamic_thresholding=dynamic_thresholding,model=model,qualityToggle = qualityToggle == 1, ucPreset = ucPreset , noise = extra_noise, image = image, strength= p.denoising_strength,extra_noise_seed = seed if p.subseed_strength <= 0 else int(p.all_subseeds[i]), overlay = False, mask = self.mask if inpaint_mode!=1 else None,legacy_v3_extend=legacy_v3_extend, reference_image=self.reference_image,reference_information_extracted=self.reference_information_extracted,reference_strength=self.reference_strength,n_samples=n_samples,variety=variety,skip_cfg_above_sigma=skip_cfg_above_sigma,deliberate_euler_ancestral_bug=deliberate_euler_ancestral_bug,prefer_brownian=prefer_brownian, characterPrompts=chars,text_tag=text_tag,legacy_uc=legacy_uc,normalize_reference_strength_multiple=self.normalize_reference_strength_multiple, director_reference_images = self.cref_image, director_reference_style=self.cref_style, director_reference_secondary_strength_values = 1.0 - self.cref_fidel )
        
        
        shared.state.job_count = p.n_iter
        shared.state.job_no = 0
        shared.state.sampling_step = 0
        shared.state.current_image_sampling_step = 0
        shared.state.sampling_steps = p.steps

        while len(self.images) < p.n_iter * p.batch_size and not shared.state.interrupted:
            DEBUG_LOG("Loading Images: ",len(self.images) // p.batch_size,p.n_iter, p.batch_size)
            count = len(self.images)
            self.load_image_batch(p,getparams)            
            DEBUG_LOG("Read Image:",count)
            for i in range(count, len(self.images)):
                if self.images[i] is None:
                    self.images[i] = self.init_images[i] if self.init_images and i < len(self.init_images) else Image.new("RGBA",(p.width, p.height), color = "black")
                    self.images[i].is_transient_image = True
                    self.comment(p, f"Returned Dummy Image {i}:{len(self.images)}")
                elif p.n_iter * p.batch_size > 1 and shared.opts.data.get('nai_api_save_during_batch', False):
                    DEBUG_LOG("Save Image During Batch",i)
                    nai_meta = shared.opts.data.get('nai_metadata_only', False)
                    images.save_image(self.images[i], p.outpath_samples, "", p.all_seeds[i], p.all_prompts[i], shared.opts.samples_format, info= "NovelAI" if nai_meta else self.texts[i], p=p,existing_info=self.images[i].info.copy(), pnginfo_section_name = 'Software' if nai_meta else 'parameters')
                    self.images[i].nai_batch_image_saved = True
                elif isimg2img and getattr(p,"inpaint_full_res",False) and shared.opts.data.get('nai_api_save_fragments', False):
                    images.save_image(self.images[i], p.outpath_samples, "", p.all_seeds[i], p.all_prompts[i], shared.opts.samples_format, info=self.texts[i], suffix="-nai-init-image" if self.do_local_img2img != 0 else "")
            shared.state.job_no = len(self.images) // p.batch_size
            shared.state.sampling_step = 0
            shared.state.current_image_sampling_step = 0
        
        if self.do_local_img2img == 0:
           p.nai_processed = Processed(p, self.images, p.seed, self.texts[0] if len(self.texts) >0 else "", subseed=p.all_subseeds[0], infotexts = self.texts) 
        elif self.do_local_img2img== 1 or self.do_local_img2img == 2:
            self.all_seeds = p.all_seeds.copy()
            self.all_subseeds = p.all_subseeds.copy()
            self.all_prompts = p.all_prompts.copy()
            self.all_negative_prompts = p.all_negative_prompts.copy()
            
            self.set_local(p,enable,convert_prompts,cost_limiter,nai_post,disable_smea_in_post,model,sampler,noise_schedule,dynamic_thresholding,variety,smea,cfg_rescale,skip_cfg_above_sigma,qualityToggle,ucPreset,do_local_img2img,extra_noise,inpaint_mode,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,legacy_v3_extend,augment_mode,defry,emotion,reclrLvlLo,reclrLvlHi,reclrLvlMid,reclrLvlLoOut,reclrLvlHiOut,reclrLvlAlpha,deliberate_euler_ancestral_bug,prefer_brownian,legacy_uc,normalize_reference_strength_multiple,normalize_negatives,normalize_level,cref_image,cref_style,cref_fidel,keep_mask_for_local)
                
            p.init_images = self.images.copy()
            self.include_nai_init_images_in_results=True
    
        self.images+=self.fragments
        self.texts+=self.fragment_texts
        
    def load_image_batch(self, p, getparams):    
        i = len(self.images)
        parameters = getparams(i)
        self.load_image(p,i, parameters)
        self.process_result_image(p, i)
        if shared.opts.live_previews_enable and self.images[i]!=None and p.n_iter > 1: 
            shared.state.assign_current_image(self.images[i].copy())
        if i == 0 and self.texts[i]:
            import modules.paths as paths
            with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    file.write(self.texts[i])

    def begin_request(self,p,i,parameters):    

        print(f"Requesting Image {i+1}/{p.n_iter*p.batch_size}: {p.width} x {p.height} - {p.steps} steps.")
        if shared.opts.data.get('nai_query_logging', False):     
            print(nai_api.shorten(parameters))

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
            if result.status_code == 200 and result.files:
                if isinstance(result.files[0], Image.Image):
                    gentime = max(gentime, nai_api.tryfloat(result.files[0].info.get('Generation time',0),0))
            else:
                err = f'Request {idx + i + 1}/{p.n_iter*p.batch_size}: {result.message}'                
                self.comment(p,err)
                if result.status_code < 0: errors.display(result, "NAI Image Request", full_traceback= True)
                has_error = True
            retry_count += result.retry_count
                
        if retry_count>0: msg = f'{msg} Retries: {retry_count:.2f}s'
        if gentime>=0.01: msg = f'{msg} Generation time: {gentime:.2f}s'
        
        if not has_error and idx == 0: self.comment(p,msg)
        return has_error
        
    def load_image(self, p, i, parameters, batch_size = 1):    
        key = self.get_api_key()        
        self.begin_request(p,i,parameters)
        
        if shared.opts.live_previews_enable and not self.augment_mode and self.isV4 and shared.opts.nai_api_preview:
            def preview(img, step):            
                shared.state.sampling_step = step
                if step - shared.state.current_image_sampling_step >= shared.opts.show_progress_every_n_steps and shared.opts.show_progress_every_n_steps != -1:
                    shared.state.assign_current_image(Image.open(BytesIO(img)))
                    shared.state.current_image_sampling_step = step
            result = nai_api.STREAM(key, parameters, preview ,timeout = nai_api.get_timeout(shared.opts.nai_api_timeout,p.width,p.height,p.steps), attempts = shared.opts.nai_api_retry , wait_on_429 = shared.opts.nai_api_wait_on_429 , wait_on_429_time = shared.opts.nai_api_wait_on_429_time)            
        else:
            result = nai_api.POST(key, parameters,timeout = nai_api.get_timeout(shared.opts.nai_api_timeout,p.width,p.height,p.steps), attempts = 1 if self.augment_mode else shared.opts.nai_api_retry , wait_on_429 = shared.opts.nai_api_wait_on_429 , wait_on_429_time = shared.opts.nai_api_wait_on_429_time, url = nai_api.NAI_AUGMENT_URL if self.augment_mode else nai_api.NAI_IMAGE_URL)
                
        if self.request_complete(p,i,result): 
            self.images.append(None)
            self.texts.append("")
            return            
        image = result.files        
        for ii in range(len(image)):
            self.images.append(image[ii])
            self.texts.append(self.infotext(p,i+ii if ii < batch_size else i))
        DEBUG_LOG("Read Image:",i)

    def process_result_image(self, p, i):
        def add_fragment(image = None):
            self.fragments.append(image or self.images[i])
            self.fragment_texts.append(self.texts[i])
        original = self.images[i]
        if original is None: return
        original.is_original_image = True
        if self.crop is not None:
            if shared.opts.data.get('nai_api_all_images', False): add_fragment()
            over = p.init_images[i % len(p.init_images)]            
            image = over.copy()
            image.paste(images.resize_image(1, self.images[i].convert('RGB'), self.crop[2]-self.crop[0], self.crop[3]-self.crop[1]), (self.crop[0], self.crop[1]))
            image = Image.composite(over, image, self.overlay_mask).convert('RGB')
            image.original_nai_image = original
            self.images[i] = image
        elif self.mask is not None:
            if shared.opts.data.get('nai_api_all_images', False): add_fragment()
            image = Image.composite(self.init_images[i % len(self.init_images)], self.images[i].convert('RGB'), self.overlay_mask).convert('RGB')
            image.original_nai_image = original
            self.images[i] = image

def maskblur(image, val):
    size = 2 * int(2.5 * val + 0.5) + 1
    ar = cv2.GaussianBlur(np.array(image), (size, size), val)
    return Image.fromarray(ar.astype(np.uint8))
def maskblur2(image, val, iterations=1):
    for i in range(iterations):
        image = image.filter(ImageFilter.BoxBlur(val))
    return image
def clip(image,val):
    ar = np.array(image)
    ar = np.where(ar>=val,255,0)
    return Image.fromarray(ar.astype(np.uint8))
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