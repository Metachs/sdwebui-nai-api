from modules import scripts, script_callbacks, shared, extra_networks, ui
import gradio as gr

from modules.ui_components import ToolButton
from modules.processing import process_images,apply_overlay
from modules.processing import Processed

from scripts import nai_api
from scripts.nai_api import NAIGenParams
from scripts import nai_script

import modules.images as images
from PIL import Image, ImageFilter, ImageOps
from modules import masking
import numpy as np


NAIv1 = "nai-diffusion"
NAIv1c = "safe-diffusion"
NAIv1f = "nai-diffusion-furry"
NAIv2 = "nai-diffusion-2"
NAIv3 = "nai-diffusion-3"

hashdic = {}    

PREFIX = 'NAI'

class NAIGENScriptText(nai_script.NAIGENScript):

    def __init__(self):
        super().__init__()    
        self.NAISCRIPTNAME = "NAI API Generation"    
        #self.query_batch_size = 1
        
    def title(self):
        return self.NAISCRIPTNAME

    def show(self, is_img2img):
        if is_img2img: return False
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label=self.NAISCRIPTNAME, open=False):
            with gr.Row(variant="compact"):
                enable = gr.Checkbox(value=False, label="Enable")
                hr = gr.HTML()
                refresh = ToolButton(ui.refresh_symbol)
                refresh.click(fn= lambda: self.connect_api(), inputs=[], outputs=[enable,hr])
            with gr.Row(variant="compact"):
                model = gr.Dropdown(label= "Model",value=nai_api.NAIv3,choices=nai_api.nai_models,type="value",show_label=False)
                sampler = gr.Dropdown(label="Sampler",value="Auto",choices=["Auto Sampler",*nai_api.NAI_SAMPLERS],type="value",show_label=False)
            with gr.Row(variant="compact"):
                dynamic_thresholding = gr.Checkbox(value=False, label='Decrisper (Dynamic Thresholding)',min_width=64)
                smea = gr.Radio(label="SMEA",value="Off",choices=["SMEA","DYN","Off"],type="value",show_label=False)            
            with gr.Accordion(label="Advanced", open=False):
                with gr.Row(variant="compact"):
                    cfg_rescale=gr.Slider(minimum=0.0, maximum=1.0, step=0.02, label='CFG Rescale', value=0.0)
                    uncond_scale=gr.Slider(minimum=0.0, maximum=1.5, step=0.05, label='Uncond Scale', value=1.0,)
                    noise_schedule = gr.Dropdown(label="Schedule",value="recommended",choices=["recommended","exponential","polyexponential","karras","native"],type="value")
            with gr.Accordion(label="Options", open=False):
                with gr.Row(variant="compact"):
                    qualityToggle = gr.Radio(value="Off", label="Quality Preset",choices=["Off","On"],type="index") 
                    #qualityToggle = gr.Checkbox(value=False, label="Quality Preset") 
                    ucPreset = gr.Radio(label="Negative Preset",value="None",choices=["Heavy","Light","None"],type="index")           
                    convert_prompts = gr.Dropdown(label="Convert Prompts for NAI ",value="Auto",choices=["Auto","Never","Always"])
                    cost_limiter = gr.Checkbox(value=True, label="Force Opus Free Gen Size/Step Limit")
                    nai_post = gr.Checkbox(value=True, label="Use NAI for Post Processing (ADetailer)")
        def on_enable(e,h):
            if e and not self.api_connected:
                return self.connect_api()
            return e,h
            
        if not self.skip_checks():
            enable.change(fn=on_enable, inputs=[enable,hr], outputs=[enable,hr])

        self.infotext_fields = [
            (enable, f'{PREFIX} '+ 'enable'),
            (sampler, f'{PREFIX} '+ 'sampler'),
            (noise_schedule, f'{PREFIX} '+ 'noise_schedule'),
            (dynamic_thresholding, f'{PREFIX} '+ 'dynamic_thresholding'),
            (model, f'{PREFIX} '+ 'model'),
            (smea, f'{PREFIX} '+ 'smea'),
            (uncond_scale, f'{PREFIX} '+ 'uncond_scale'),
            (cfg_rescale, f'{PREFIX} '+ 'cfg_rescale'),
        ]
        
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)
            
        return [enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset]
    
    def patched_process(self,p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,**kwargs):
        if not enable: self.disabled=True
        if self.disabled: return 
        
        if not self.check_api_key():
            self.fail(p,"Invalid NAI Key")
            return
        self.do_nai_post=nai_post
            
        self.setup_sampler_name(p, sampler)
        
        if cost_limiter: self.limit_costs(p)
        
        self.adjust_resolution(p)
        #self.query_batch_size = p.batch_size
        
        if not p.enable_hr: 
            p.disable_extra_networks=True
        else:
            p.n_iter = p.n_iter * p.batch_size
            p.batch_size = 1
            
        p.batch_size = p.n_iter * p.batch_size
        p.n_iter = 1

    def process_inner(self, p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset, **kwargs):
        if not enable: self.disabled=True
        if self.disabled: return 
        
        
        model = getattr(p,f'{PREFIX}_'+ 'model',model)
        smea = getattr(p,f'{PREFIX}_'+ 'smea',smea)
        sampler = getattr(p,f'{PREFIX}_'+ 'sampler',None) or self.sampler_name        
        noise_schedule = getattr(p,f'{PREFIX}_'+ 'noise_schedule',noise_schedule)
        dynamic_thresholding = getattr(p,f'{PREFIX}_'+ 'dynamic_thresholding',dynamic_thresholding)
        uncond_scale = getattr(p,f'{PREFIX}_'+ 'uncond_scale',uncond_scale)
        cfg_rescale = getattr(p,f'{PREFIX}_'+ 'cfg_rescale',cfg_rescale)       
        
        p.extra_generation_params[f'{PREFIX} '+ 'enable'] = True 
        if sampler in nai_api.NAI_SAMPLERS : p.extra_generation_params[f'{PREFIX} _sampler'] = sampler
        p.extra_generation_params[f'{PREFIX} '+ 'noise_schedule'] = noise_schedule
        p.extra_generation_params[f'{PREFIX} '+ 'dynamic_thresholding'] = dynamic_thresholding
        p.extra_generation_params[f'{PREFIX} '+ 'model'] = model
        p.extra_generation_params[f'{PREFIX} '+ 'smea'] = smea
        p.extra_generation_params[f'{PREFIX} '+ 'uncond_scale'] = uncond_scale
        p.extra_generation_params[f'{PREFIX} '+ 'cfg_rescale'] = cfg_rescale
        
        def getparams(i):   
            seed =int(p.all_seeds[i])
            prompt,neg = self.convert_to_nai(p.all_prompts[i],  p.all_negative_prompts[i], convert_prompts)
            return NAIGenParams(prompt, neg, seed=seed , width=p.width, height=p.height, scale=p.cfg_scale, sampler=sampler, steps=p.steps, noise_schedule=noise_schedule,sm= smea.lower() == "smea", sm_dyn="dyn" in smea.lower(), cfg_rescale=cfg_rescale,uncond_scale=uncond_scale ,dynamic_thresholding=dynamic_thresholding,model=model,qualityToggle = qualityToggle == 1 , ucPreset = ucPreset)
        
        self.get_batch_images(p, getparams, save_images = True , save_suffix = "" ,dohash = False, query_batch_size=1)

        p.nai_processed = Processed(p, self.images, p.seed, self.texts[0], subseed=p.all_subseeds[0], infotexts = self.texts) 


    def post_process_i2i(self, p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset, **kwargs):
        print("POSTPROCESS")
        if p.init_images is None or len(p.init_images) == 0:
            return None
        print("POSTPROCESS2" )
        self.setup_sampler_name(p, sampler)
        if cost_limiter: self.limit_costs(p)
        self.adjust_resolution(p)
        p.disable_extra_networks=True

        p.batch_size = p.n_iter * p.batch_size
        p.n_iter = 1        
        image_mask = p.image_mask
        
        crop = None
        if image_mask is not None: 
            if p.inpaint_full_res:
                mask = image_mask.convert('L')
                crop = masking.expand_crop_region(masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding), p.width, p.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop
                image_mask = images.resize_image(2, mask.crop(crop), p.width, p.height)
                paste_to = (crop[0], crop[1], crop[2]-crop[0], crop[3]-crop[1])                
                init_masked=[]
                for i in range(len(p.init_images)):
                    image = p.init_images[i]                
                    image_masked = Image.new('RGBa', (image.width, image.height))
                    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(p.image_mask.convert('L')))
                    init_masked.append(image_masked.convert('RGBA'))
            # elif img_resize_mode < 3:
                # image_mask = images.resize_image(img_resize_mode, image_mask, p.width, p.height)


        def getparams(i):
            seed =int(p.all_seeds[i])
            
            image= p.init_images[len(p.init_images) % p.batch_size]
            
            if crop is not None:
                image = image.crop(crop)
                image = images.resize_image(2, image, p.width, p.height)
            # elif image is not None and img_resize_mode < 3:
                # image = images.resize_image(img_resize_mode, image, p.width, p.height)
                
            prompt,neg = self.convert_to_nai(p.all_prompts[i],  p.all_negative_prompts[i], convert_prompts)
            
            return NAIGenParams(prompt, neg, seed=seed , width=p.width, height=p.height, scale=p.cfg_scale, sampler = self.sampler_name, steps=p.steps, noise_schedule=noise_schedule,sm=smea.lower()=="smea", sm_dyn="dyn" in smea.lower(), cfg_rescale=cfg_rescale,uncond_scale=uncond_scale ,dynamic_thresholding=dynamic_thresholding,model=model,qualityToggle = 0, ucPreset = 2 , noise = 0, image = image, strength= p.denoising_strength,overlay=True, mask = image_mask)        
                
        self.images = []
        self.texts = []
        self.hashes = []
        
        self.get_batch_images(p, getparams, save_images = False ,dohash = False, query_batch_size=1)        
        
        if crop is not None:
            for i in range(len(self.images)):
                image = apply_overlay(self.images[i], paste_to, 0, init_masked)
                images.save_image(image, p.outpath_samples, "", p.all_seeds[i], p.all_prompts[i], shared.opts.samples_format, info=self.texts[i])
                self.images[i] = image
                
        return Processed(p, self.images, p.seed, self.texts[0], subseed=p.subseed, infotexts = self.texts) 
        