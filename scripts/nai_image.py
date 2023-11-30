from modules import scripts, script_callbacks, shared, extra_networks, ui
import gradio as gr

from modules.ui_components import ToolButton
from modules.ui_common import plaintext_to_html

from scripts import nai_api
from scripts.nai_api import NAIGenParams 
from scripts import nai_script

import modules.images as images
from modules.processing import process_images,apply_overlay
from modules.processing import Processed
from PIL import Image, ImageFilter, ImageOps
from modules import masking
import numpy as np

PREFIX = 'NAI'

class NAIGENScriptText(nai_script.NAIGENScript):

    def __init__(self):
        super().__init__()    
        self.NAISCRIPTNAME = "NAI API Generation I2I"    
        #self.query_batch_size = 1
        self.width = 0
        self.height = 0
        self.mask = None
        self.cfg = 0
        self.steps = 0
        self.strength = 0
        
    def title(self):
        return self.NAISCRIPTNAME

    def show(self, is_img2img):
        if not is_img2img: return False
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        
        with gr.Accordion(label=self.NAISCRIPTNAME, open=False):
            with gr.Row(variant="compact"):
                enable = gr.Checkbox(value=False, label="Enable")
                hr = gr.HTML()
                refresh = ToolButton(ui.refresh_symbol)
                refresh.click(fn= lambda: self.connect_api(), inputs=[], outputs=[enable,hr])
            with gr.Row(variant="compact"):
                modes = ["One Pass: NAI Img2Img/Inpaint", "Two Pass: NAI Txt2Img > Local Img2Img (Ignore Source Image)","Two Pass: NAI Img2Img/Inpaint > Local Img2Img"]
                do_local = gr.Dropdown(value=modes[0],choices= modes,type="index", label="Mode")
            with gr.Row(variant="compact"):
                model = gr.Dropdown(label= "Model",value=nai_api.NAIv3,choices=nai_api.nai_models,type="value",show_label=False)
                sampler = gr.Dropdown(label="Sampler",value="Auto",choices=["Auto",*nai_api.NAI_SAMPLERS],type="value",show_label=False)
            with gr.Row(variant="compact"):
                dynamic_thresholding = gr.Checkbox(value=False, label='Decrisper (Dynamic Thresholding)',min_width=64)
                smea = gr.Radio(label="SMEA",value="Off",choices=["SMEA","DYN","Off"],type="value",show_label=False)            
            with gr.Row(variant="compact"):
                img_resize_mode = gr.Dropdown(label='Resize Mode',  choices=["Resize","Crop","Fill","None (NAI Resize)"], value = "Resize", type="index")
                extra_noise=gr.Slider(minimum=0.0, maximum=1.0 ,step=0.01, label='Noise', value=0.0)
                add_original_image = gr.Checkbox(value=True, label='Inpaint: Overlay Image')            
            with gr.Accordion(label="Advanced", open=False):
                with gr.Row(variant="compact"):
                    cfg_rescale=gr.Slider(minimum=0.0, maximum=1.0, step=0.02, label='CFG Rescale', value=0.0)
                    uncond_scale=gr.Slider(minimum=0.0, maximum=1.5, step=0.05, label='Uncond Scale', value=1.0)
                    noise_schedule = gr.Dropdown(label="Schedule",value="recommended",choices=["recommended","exponential","polyexponential","karras","native"],type="value")
            with gr.Accordion(label='Local Second Pass Overrides: Ignored if 0', open=False ):                    
                with gr.Row(variant="compact"):
                    nai_resolution_scale=gr.Slider(minimum=0.0, maximum=4.0, step=0.05, label='Scale', value=1.0)
                    nai_cfg=gr.Slider(minimum=0.0, maximum=30, step=0.05, label='CFG', value=0.0)
                    nai_steps=gr.Slider(minimum=0, maximum=150, step=1, label='Steps', value=0)
                    nai_denoise_strength=gr.Slider(minimum=0.0, maximum=1.0,step=0.01, label='Denoise strength', value=0.0)
                    keep_mask_for_local = gr.Checkbox(value=False, label="Keep inpaint mask for both passes")
            with gr.Accordion(label="Options", open=False):
                with gr.Row(variant="compact"):
                    qualityToggle = gr.Radio(value="Off", label="Quality Preset",choices=["Off","On"],type="index") 
                    ucPreset = gr.Radio(label="Negative Preset",value="None",choices=["Heavy","Light","None"],type="index")           
                    convert_prompts = gr.Dropdown(label="Convert Prompts for NAI ",value="Auto",choices=["Auto","Never","Always"])
                    cost_limiter = gr.Checkbox(value=True, label="Force Opus Free Gen Size/Step Limit")
                    nai_post = gr.Checkbox(value=True, label="Use NAI for Inpainting with ADetailer")
        def on_enable(e,h):
            if e and not self.api_connected: return self.connect_api()
            return e,h
            
        if not self.skip_checks():
            enable.change(fn=on_enable, inputs=[enable,hr], outputs=[enable,hr])

        self.infotext_fields = [
            (enable, f'{PREFIX} enable I'),
            (sampler, f'{PREFIX} sampler'),
            (noise_schedule, f'{PREFIX} noise_schedule'),
            (dynamic_thresholding, f'{PREFIX} dynamic_thresholding'),
            (model, f'{PREFIX} '+ 'model'),
            (smea, f'{PREFIX} '+ 'smea'),
            (uncond_scale, f'{PREFIX} '+ 'uncond_scale'),
            (cfg_rescale, f'{PREFIX} '+ 'cfg_rescale'),
            
            (keep_mask_for_local, f'{PREFIX} '+ 'keep_mask_for_local'),
            (nai_denoise_strength, f'{PREFIX} '+ 'nai_denoise_strength'),
            (nai_steps, f'{PREFIX} '+ 'nai_steps'),
            (nai_cfg, f'{PREFIX} '+ 'nai_cfg'),
            (add_original_image, f'{PREFIX} '+ 'add_original_image'),
            (extra_noise, f'{PREFIX} '+ 'extra_noise'),
        ]
        
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)
            
        return [enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local]
        
    def can_init_script(self,p):
        return hasattr(p,"init_images")
        
    def patched_process(self,p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local,**kwargs):
        
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
        self.width = p.width
        self.height= p.height
        self.cfg = p.cfg_scale
        self.steps = p.steps
        self.strength = p.denoising_strength
        self.mask = p.image_mask
        
        if do_local != 0:
            self.set_local(p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local)
        else:
            p.disable_extra_networks=True
            
        self.images = []
        self.hashes = []
        
        p.batch_size = p.n_iter * p.batch_size
        p.n_iter = 1
        
    def restore_local(self,p):
        p.width = self.width
        p.height= self.height
        p.cfg_scale = self.cfg
        p.steps = self.steps
        p.image_mask = self.mask
        p.denoising_strength = self.strength
        
    def set_local(self,p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local):    
        if nai_resolution_scale> 0:
            p.width = int(p.width * nai_resolution_scale)
        p.height = int(p.height * nai_resolution_scale)
        if nai_cfg > 0: p.cfg_scale = nai_cfg
        if nai_steps > 0: p.steps = nai_steps
        if nai_denoise_strength > 0: p.denoising_strength = nai_denoise_strength
        if not keep_mask_for_local and do_local == 2: p.image_mask = None

    def process_inner(self,p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local,**kwargs):
        if not enable: self.disabled=True
        if self.disabled: return 
        
        if do_local != 0: self.restore_local(p)
        
        model = getattr(p,f'{PREFIX}_'+ 'model',model)
        smea = getattr(p,f'{PREFIX}_'+ 'smea',smea)
        sampler = getattr(p,f'{PREFIX}_'+ 'sampler',None) or self.sampler_name        
        noise_schedule = getattr(p,f'{PREFIX}_'+ 'noise_schedule',noise_schedule)
        dynamic_thresholding = getattr(p,f'{PREFIX}_'+ 'dynamic_thresholding',dynamic_thresholding)
        uncond_scale = getattr(p,f'{PREFIX}_'+ 'uncond_scale',uncond_scale)
        cfg_rescale = getattr(p,f'{PREFIX}_'+ 'cfg_rescale',cfg_rescale)        
        extra_noise = getattr(p,f'{PREFIX}_'+ 'extra_noise',extra_noise)        
        add_original_image = getattr(p,f'{PREFIX}_'+ 'add_original_image',add_original_image)        
        
        p.extra_generation_params[f'{PREFIX} enable I'] = True
        if sampler.lower() != "auto": p.extra_generation_params[f'{PREFIX} sampler'] = sampler
        p.extra_generation_params[f'{PREFIX} noise_schedule'] = noise_schedule
        p.extra_generation_params[f'{PREFIX} dynamic_thresholding'] = dynamic_thresholding
        p.extra_generation_params[f'{PREFIX} '+ 'smea'] = smea
        p.extra_generation_params[f'{PREFIX} '+ 'uncond_scale'] = uncond_scale
        p.extra_generation_params[f'{PREFIX} '+ 'cfg_rescale'] = cfg_rescale
        
        p.extra_generation_params[f'{PREFIX} '+ 'keep_mask_for_local'] = keep_mask_for_local
        p.extra_generation_params[f'{PREFIX} '+ 'nai_denoise_strength'] = nai_denoise_strength
        p.extra_generation_params[f'{PREFIX} '+ 'nai_steps'] = nai_steps
        p.extra_generation_params[f'{PREFIX} '+ 'nai_cfg'] = nai_cfg
        p.extra_generation_params[f'{PREFIX} '+ 'nai_resolution_scale'] = nai_resolution_scale
        p.extra_generation_params[f'{PREFIX} '+ 'add_original_image'] = add_original_image
        p.extra_generation_params[f'{PREFIX} '+ 'extra_noise'] = extra_noise
        p.extra_generation_params[f'{PREFIX} '+ 'img_resize_mode'] = img_resize_mode
        
            
        
            
        extra_noise = max(getattr(p,"extra_noise",0) , extra_noise)        
        
        if do_local == 0 and p.denoising_strength == 0:
            for i in range(len(p.init_images)):
                self.images.append(p.init_images[i])
                self.texts.append(self.infotext(p,i))

            p.nai_processed = Processed(p, self.images, p.seed, self.texts[0], subseed=p.all_subseeds[0], infotexts = self.texts) 
            return
        
        image_mask = p.image_mask or self.mask        
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
            elif img_resize_mode < 3:
                image_mask = images.resize_image(img_resize_mode, image_mask, p.width, p.height)

        def getparams(i):
            seed =int(p.all_seeds[i])
            
            image= None if (do_local == 1 or p.init_images is None or len(p.init_images) == 0) else  p.init_images[len(p.init_images) % min(p.batch_size, len (p.init_images))]
            
            if crop is not None:
                image = image.crop(crop)
                image = images.resize_image(2, image, p.width, p.height)
            elif image is not None and img_resize_mode < 3:
                image = images.resize_image(img_resize_mode, image, p.width, p.height)
            
            prompt,neg = self.convert_to_nai(p.all_prompts[i],  p.all_negative_prompts[i], convert_prompts)
            
            return NAIGenParams(prompt, neg, seed=seed , width=p.width, height=p.height, scale=p.cfg_scale, sampler = self.sampler_name, steps=p.steps, noise_schedule=noise_schedule,sm=smea.lower()=="smea", sm_dyn="dyn" in smea.lower(), cfg_rescale=cfg_rescale,uncond_scale=uncond_scale ,dynamic_thresholding=dynamic_thresholding,model=model,qualityToggle = qualityToggle == 1, ucPreset = ucPreset , noise = extra_noise, image = image, strength= p.denoising_strength,overlay=add_original_image, mask = image_mask)        
        suffix ="-nai-init-image" if do_local > 0 else ""       
        
        
        self.get_batch_images(p, getparams, save_images = not p.inpaint_full_res or shared.opts.data.get('nai_api_save_fragments', False) , save_suffix =suffix ,dohash = False, query_batch_size=1)
        
        if crop is not None:
            fragments = self.images.copy() if shared.opts.data.get('nai_api_all_images', False) else None
            for i in range(len(self.images)):
                image = apply_overlay(self.images[i], paste_to, 0, init_masked)
                self.images[i] = image
            if fragments is not None:
                self.images+=fragments
                self.texts*=2

        if do_local == 0:
           p.nai_processed = Processed(p, self.images, p.seed, self.texts[0], subseed=p.all_subseeds[0], infotexts = self.texts) 
        else:
            
            self.all_seeds = p.all_seeds.copy()
            self.all_subseeds = p.all_subseeds.copy()
            self.all_prompts = p.all_prompts.copy()
            self.all_negative_prompts = p.all_negative_prompts.copy()
            
            self.set_local(p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local)
                
            p.init_images = self.images
            self.include_nai_init_images_in_results=True
            
        
    def post_process_i2i(self,p,enable,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local,**kwargs):
        return self.do_post_process(p,convert_prompts,cost_limiter,nai_post,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,add_original_image)
