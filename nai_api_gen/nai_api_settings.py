from modules import scripts, script_callbacks,shared
import gradio as gr
from nai_api_gen import nai_api

def script_setup():
    script_callbacks.on_ui_settings(on_ui_settings)

def on_ui_settings():
    section = ('nai_api', "NAI API Generator")
    def addopt(n,o):
        if not n in shared.opts.data_labels:
            shared.opts.add_option(n,o)
            
    addopt('nai_api_key', shared.OptionInfo('', "NAI API Key - See https://docs.sillytavern.app/usage/api-connections/novelai/ ", gr.Textbox, section=section))
    
    addopt('nai_api_skip_checks', shared.OptionInfo(False, "Skip NAI account/subscription/Anlas checks.",gr.Checkbox, section=section))
    
    addopt('nai_api_preview', shared.OptionInfo(True, "Stream preview images during generation.",gr.Checkbox, section=section))
    
    addopt('nai_api_convertImportedWeights', shared.OptionInfo(True, "Convert NAI weights to A1111 when importing NAI prompts.",gr.Checkbox, section=section))
    addopt('nai_api_use_numeric_emphasis', shared.OptionInfo(True, "Use numeric emphasis (1.2:: ... ::) when converting sdwebui prompt weights for NAI.",gr.Checkbox, section=section))
    
    addopt('nai_api_recommended_schedule_overrides', shared.OptionInfo('k_dpmpp_2m:exponential k_dpmpp_sde:native  k_dpmpp_2s_ancestral:native', f"Override the recommended Noise Schedule (default is karras, formerly native). Format - 'sampler:schedule' pairs separated by spaces. Samplers - [{', '.join(nai_api.NAI_SAMPLERS)}] Schedules - [{', '.join(nai_api.noise_schedules)}] .",gr.Textbox,section=section))    
    
    addopt('nai_api_png_info', shared.OptionInfo( 'NAI Only', "Stealth PNG Info - Write Stealth PNG info for NAI images only (required to emulate NAI), or All Images",gr.Radio, {"choices": ['NAI Only', 'All Images'] }, section=section))
            
    addopt('nai_api_png_info_read', shared.OptionInfo(True, "Read Stealth PNG Info from images",gr.Checkbox, section=section))
        
    addopt('nai_api_vibe_v4_count', shared.OptionInfo(4, "Maximum Number of Vibe Transfer Images for v4, opus limit=4, Requires Restart",gr.Slider, {"minimum":  1, "maximum": 32, "step": 1}, section=section))

    addopt('nai_api_use_v4_for_v3', shared.OptionInfo(True, "Allow using v4 Vibe interface with other models",gr.Checkbox, section=section))

    addopt('nai_api_vibe_v4_directory', shared.OptionInfo('', "Directory where v4 Vibe Image Encodings are cached, uses 'vibe_encodings' in this extension's directory by default.", gr.Textbox, section=section))
    
    addopt('nai_api_vibe_v4_encoding_directory', shared.OptionInfo('', "Directory where imageless v4 Vibe Encodings are cached, uses same directory as Image encodings if unspecified.", gr.Textbox, section=section))
    
    addopt('nai_api_vibe_v4_preview_directory', shared.OptionInfo('', "Directory for full size v4 preview images, does NOT store encodings, only provides a way to view image with a custom name for loading into sdwebui.", gr.Textbox, section=section))
    
    
    
    addopt('nai_api_vibe_always_encode', shared.OptionInfo(False, "Always encode Vibe reference images regardless of cost limiter.",gr.Checkbox, section=section))
    
    
    addopt('nai_api_vibe_count', shared.OptionInfo(4, "Maximum Number of Vibe Transfer Images, Requires Restart",gr.Slider, {"minimum":  1, "maximum": 32, "step": 1}, section=section))
    
    
    addopt('nai_api_vibe_pre', shared.OptionInfo( 'NAI', "Preprocess Vibe Images - NAI: Use NAI's resizing method, (Resizes images to 448x448, padding with black, browser-dependendent low quality scaling). None: Send original images, this changes results as NAI's servers seem to use a different scaling implementation. Scale: Scale image to fit inside 448x448 without padding (Lanczos, higher quality than the NAI method).",gr.Radio, {"choices": ['NAI','Scale','None']  }, section=section))

    addopt('nai_api_delay', shared.OptionInfo(0, "Minimum Time in seconds between successive API requests, may reduce Too Many Request errors.",gr.Slider, {"minimum":  0, "maximum": 32, "step": 1}, section=section))
    
    addopt('nai_api_delay_rand', shared.OptionInfo(0, "Random amount of time to wait between requests, adds a random value between 0 and n seconds to Minimum Time. May help avoid being incorrectly flagged as a bot.",gr.Slider, {"minimum":  0, "maximum": 32, "step": 1}, section=section))
    
    
    addopt('nai_api_timeout', shared.OptionInfo(120 , "Request Timeout - Time in seconds to wait before giving up on a request",gr.Slider, {"minimum":  30, "maximum": 360, "step": 1}, section=section))
    
    addopt('nai_api_retry', shared.OptionInfo(2, "Request Retry Attempts - Number of times to retry a failed request before giving up",gr.Slider, {"minimum": 0, "maximum": 12, "step": 1}, section=section))   
    
    
    addopt('nai_api_wait_on_429', shared.OptionInfo(30, "Maximum total time to wait before failing after receiving a 429 error (Too many requests).",gr.Slider, {"minimum": 0, "maximum": 3600, "step": 1}, section=section))
    
    addopt('nai_api_wait_on_429_time', shared.OptionInfo(5, "Time to wait between retry attempts after receiving a 429 error (Too many requests).",gr.Slider, {"minimum": 0, "maximum": 120, "step": 1}, section=section))
    
    
    addopt('nai_api_default_sampler', shared.OptionInfo('k_euler', "Fallback Sampler: Used when the NAI sampler is set to Auto and the primary Sampling method doesn't match any sampler available in NAI .",gr.Radio, {"choices": nai_api.NAI_SAMPLERS }, section=section))
    
    addopt('nai_metadata_only', shared.OptionInfo(False, "Save only NAI metadata instead of full local metadata. Local metadata can be useful for extensions, but contains irrelevant info.",gr.Checkbox, section=section))
    
    addopt('nai_api_include_original', shared.OptionInfo(True, "Include Original NAI images before Post-Processing in results.",gr.Checkbox, section=section))
    addopt('nai_api_save_original', shared.OptionInfo(False, "Save Original NAI images before Post-Processing.",gr.Checkbox, section=section))
    
    
    addopt('nai_api_all_images', shared.OptionInfo(False, "Include all images generated by NAI in the results.",gr.Checkbox, section=section))
    addopt('nai_api_save_fragments', shared.OptionInfo(False, "Save File Fragments generated by NAI while inpainting.",gr.Checkbox, section=section))
    
    addopt('nai_query_logging', shared.OptionInfo(False, "Log NAI API Queries to console.",gr.Checkbox, section=section))

    addopt('nai_verbose_logging', shared.OptionInfo(False, "Enable debug logging. IE Dump a bunch of garbage to the console.",gr.Checkbox, section=section))

    addopt('nai_alt_action_tag', shared.OptionInfo('::', "Alternate Action Tag - When used in character prompts, will be replaced with '#'. Workaround for NAI's action tag syntax ('target#pointing') conflicting with sdwebui's Prompt Comment feature, which uses '#' to start comments, breaking '#' action tags. Leave empty to disable.", gr.Text, section=section))
        
def get_recommended_schedule(sampler_name):
    dic = {}
    for kv in shared.opts.nai_api_recommended_schedule_overrides.lower().split():
        if ':' in kv: 
            k,v,*_ = kv.split(':')
            dic[k]=v       
    return dic.get(sampler_name, 'karras')

def noise_schedule_selected(sampler,noise_schedule):
    noise_schedule=noise_schedule.lower()
    sampler=sampler.lower()
    
    if noise_schedule not in nai_api.noise_schedules or sampler == "ddim": return False
    
    return noise_schedule != get_recommended_schedule(sampler)

def get_set_noise_schedule(sampler,noise_schedule):
    if sampler == "ddim": return ""
    if noise_schedule_selected(sampler, noise_schedule): return noise_schedule
    return nai_api.noise_schedule_selections[0]
    get_recommended_schedule

def DEBUG_LOG(*args,**kwargs): 
    if shared.opts.data.get('nai_verbose_logging', False): 
        print(*args,**kwargs)
        
        