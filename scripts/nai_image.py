from modules import scripts, script_callbacks, shared, extra_networks, ui
import gradio as gr

from modules.ui_components import ToolButton
from modules.ui_common import plaintext_to_html

from scripts import nai_api
from scripts.nai_api import NAIGenParams 
from scripts import nai_script

import modules.images as images
from modules.processing import process_images
from modules.processing import Processed, StableDiffusionProcessingTxt2Img,StableDiffusionProcessingImg2Img,create_infotext

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
        self.NAISCRIPTNAME = "NAI API Generation I2I"    
        self.disable = False
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
                sampler = gr.Dropdown(label="Sampler",value="Auto",choices=["Auto Sampler",*nai_api.NAI_SAMPLERS],type="value",show_label=False)
            with gr.Row(variant="compact"):
                dynamic_thresholding = gr.Checkbox(value=False, label='Decrisper (Dynamic Thresholding)',min_width=64)
                smea = gr.Radio(label="SMEA",value="Off",choices=["SMEA","DYN","Off"],type="value",show_label=False)            
            with gr.Row(variant="compact"):
                img_resize_mode = gr.Dropdown(label='Resize Mode',  choices=["Resize","Crop","Fill","None (NAI Resize)"], value = "Resize", type="index")
                extra_noise=gr.Slider(minimum=0.0, step=0.01, label='noise', value=0.0)
                add_original_image = gr.Checkbox(value=False, label='Inpaint: Overlay Image')            
            with gr.Accordion(label="Advanced", open=False):
                with gr.Row(variant="compact"):
                    cfg_rescale=gr.Slider(minimum=0.0, maximum=1.0, step=0.02, label='CFG Rescale', value=0.0)
                    uncond_scale=gr.Slider(minimum=0.0, maximum=1.5, step=0.05, label='Uncond Scale', value=1.0,)
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
                    #qualityToggle = gr.Checkbox(value=False, label="Quality Preset") 
                    ucPreset = gr.Radio(label="Negative Preset",value="None",choices=["Heavy","Light","None"],type="index")           
                    convert_prompts = gr.Dropdown(label="Convert Prompts for NAI ",value="Auto",choices=["Auto","Never","Always"])
                    cost_limiter = gr.Checkbox(value=True, label="Force Opus Free Gen Size/Step Limit")
        def on_enable(e,h):
            if e and not self.api_connected: return self.connect_api()
            return e,h
            
        if not self.skip_checks():
            enable.change(fn=on_enable, inputs=[enable,hr], outputs=[enable,hr])

        self.infotext_fields = [
            (enable, f'{PREFIX} enable'),
            (sampler, f'{PREFIX} sampler'),
            (noise_schedule, f'{PREFIX} noise_schedule'),
            (dynamic_thresholding, f'{PREFIX} dynamic_thresholding'),
            (model, f'{PREFIX} '+ 'model'),
            (smea, f'{PREFIX} '+ 'smea'),
            (uncond_scale, f'{PREFIX} '+ 'uncond_scale'),
            (cfg_rescale, f'{PREFIX} '+ 'cfg_rescale'),
            (do_local, f'{PREFIX} '+ 'mode'),
            
            (keep_mask_for_local, f'{PREFIX} '+ 'keep_mask_for_local'),
            (nai_denoise_strength, f'{PREFIX} '+ 'nai_denoise_strength'),
            (nai_steps, f'{PREFIX} '+ 'nai_steps'),
            (nai_cfg, f'{PREFIX} '+ 'nai_cfg'),
            (nai_resolution_scale, f'{PREFIX} '+ 'nai_resolution_scale'),
            (add_original_image, f'{PREFIX} '+ 'add_original_image'),
            (extra_noise, f'{PREFIX} '+ 'extra_noise'),
            (img_resize_mode, f'{PREFIX} '+ 'img_resize_mode'),
        ]
        
        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)
            
        return [enable,convert_prompts,cost_limiter,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local]
        
    def patched_process(self,p,enable,convert_prompts,cost_limiter,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local,**kwargs):
        if not enable: return
        self.setup_old(p,enable,convert_prompts,cost_limiter,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local)
        
    def setup_old(self,p,enable,convert_prompts,cost_limiter,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local):    
        
        if not self.check_api_key():
            print("Invalid NAI Key")
            self.disabled = True            
            return
        
        self.setup_sampler_name(p, sampler)
        
        self.disabled = False

        if cost_limiter: self.limit_costs(p)
        self.adjust_resolution(p)
        #self.query_batch_size = p.batch_size        
        self.width = p.width
        self.height= p.height
        self.cfg = p.cfg_scale
        self.steps = p.steps
        self.strength = p.denoising_strength
        self.mask = p.image_mask
        
        if do_local:
            if p.n_iter> 1: print(f"{self.NAISCRIPTNAME} does not currently support iteration in 2 pass mode")
            
            self.set_local(p,enable,convert_prompts,cost_limiter,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local)
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

        
    def set_local(self,p,enable,convert_prompts,cost_limiter,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local):    
        if nai_resolution_scale> 0:
            p.width = int(p.width * nai_resolution_scale)
        p.height = int(p.height * nai_resolution_scale)
        if nai_cfg > 0: p.cfg_scale = nai_cfg
        if nai_steps > 0: p.steps = nai_steps
        if nai_denoise_strength > 0: p.denoising_strength = nai_denoise_strength
        if not keep_mask_for_local and do_local == 2: p.image_mask = None

    def process_inner(self,p,enable,convert_prompts,cost_limiter,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local,**kwargs):
    
        if not enable or self.disable: return         
        self.restore_local(p)
        p.extra_generation_params[f'{PREFIX} enable'] = True
        if sampler.lower() != "auto": p.extra_generation_params[f'{PREFIX} sampler'] = sampler
        p.extra_generation_params[f'{PREFIX} noise_schedule'] = noise_schedule
        p.extra_generation_params[f'{PREFIX} dynamic_thresholding'] = dynamic_thresholding
        p.extra_generation_params[f'{PREFIX} '+ 'smea'] = smea
        p.extra_generation_params[f'{PREFIX} '+ 'uncond_scale'] = uncond_scale
        p.extra_generation_params[f'{PREFIX} '+ 'cfg_rescale'] = cfg_rescale
        p.extra_generation_params[f'{PREFIX} '+ 'mode'] = do_local
        
        p.extra_generation_params[f'{PREFIX} '+ 'keep_mask_for_local'] = keep_mask_for_local
        p.extra_generation_params[f'{PREFIX} '+ 'nai_denoise_strength'] = nai_denoise_strength
        p.extra_generation_params[f'{PREFIX} '+ 'nai_steps'] = nai_steps
        p.extra_generation_params[f'{PREFIX} '+ 'nai_cfg'] = nai_cfg
        p.extra_generation_params[f'{PREFIX} '+ 'nai_resolution_scale'] = nai_resolution_scale
        p.extra_generation_params[f'{PREFIX} '+ 'add_original_image'] = add_original_image
        p.extra_generation_params[f'{PREFIX} '+ 'extra_noise'] = extra_noise
        p.extra_generation_params[f'{PREFIX} '+ 'img_resize_mode'] = img_resize_mode
            
        extra_noise = max(getattr(p,"extra_noise",0) , extra_noise)        
        mask = p.image_mask or self.mask
        if mask != None and img_resize_mode < 3:
            mask = images.resize_image(img_resize_mode, mask, p.width, p.height,"Lanczos")

        def getparams(i):   
            seed =int(p.all_seeds[i])
            
            image= None if do_local == 1 else  p.init_images[len(p.init_images) % p.batch_size]
            
            if image is not None and img_resize_mode < 3:
                image = images.resize_image(img_resize_mode, image, p.width, p.height,"Lanczos")
            
            prompt,neg = self.convert_to_nai(p.all_prompts[i],  p.all_negative_prompts[i], convert_prompts)
            
            return NAIGenParams(prompt, neg, seed=seed , width=p.width, height=p.height, scale=p.cfg_scale, sampler = self.sampler_name, steps=p.steps, noise_schedule=noise_schedule,sm=smea.lower()=="smea", sm_dyn="dyn" in smea.lower(), cfg_rescale=cfg_rescale,uncond_scale=uncond_scale ,dynamic_thresholding=dynamic_thresholding,model=model,qualityToggle = qualityToggle == 1, ucPreset = ucPreset , noise = extra_noise, image = image, strength= p.denoising_strength,overlay=add_original_image, mask = mask)        
        
        self.get_batch_images(p, getparams, save_images = True , save_suffix ="-nai-init-image" ,dohash = False, query_batch_size=1)        
        
        self.set_local(p,enable,convert_prompts,cost_limiter,model,sampler,noise_schedule,dynamic_thresholding,smea,cfg_rescale,uncond_scale,qualityToggle,ucPreset,do_local,extra_noise,add_original_image,nai_resolution_scale,nai_cfg,nai_steps,nai_denoise_strength,img_resize_mode,keep_mask_for_local)
            
        if not do_local:
           p.nai_processed = Processed(p, self.images, p.seed, self.texts[0], subseed=p.all_subseeds[0], infotexts = self.texts) 
        else:
            p.init_images = self.images
            