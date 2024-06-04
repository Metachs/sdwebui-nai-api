import os
from io import BytesIO
from zipfile import ZipFile
from PIL import Image 

import hashlib
import math
import re
import base64
import time
import requests

NAIv1 = "nai-diffusion"
NAIv1c = "safe-diffusion"
NAIv1f = "nai-diffusion-furry"
NAIv2 = "nai-diffusion-2"
NAIv3 = "nai-diffusion-3"
NAIv3f = "nai-diffusion-furry-3"

nai_models = [NAIv3,NAIv2,NAIv1,NAIv1c,NAIv1f,NAIv3f]

NAI_SAMPLERS = ["k_euler","k_euler_ancestral","k_dpmpp_2s_ancestral","k_dpmpp_2m","ddim","k_dpmpp_sde"]
noise_schedules = ["exponential","polyexponential","karras","native"]

noise_schedule_selections = ["recommended","exponential","polyexponential","karras","native"]

def get_headers(key):
    return {
        'accept': "*/*",
        'accept-language': "en-US,en;q=0.9,ja;q=0.8",   
        'Authorization':"Bearer "+ key,     
        'content-type': "application/json",
    }

TERMINAL_ERRORS = [401,403,404]

last_request_time = None

def POST(key,parameters, attempts = 0, timeout = 120, wait_on_429 = 0, wait_on_429_time = 5, minimum_delay = 0, wait_time_rand = 0):
    try:
        if minimum_delay + wait_time_rand > 0 and last_request_time is not None:         
            delay = ( minimum_delay + random.randint(0, wait_time_rand) ) - (time.time() - last_request_time)
            if delay > 0: time.sleep(delay)
        last_request_time = time.time()
        r = requests.post('https://image.novelai.net/ai/generate-image',headers=get_headers(key), data=parameters.encode(),timeout= timeout)
        if attempts > 0 and r is not None and r.status_code!= 200 and r.status_code not in TERMINAL_ERRORS:
            if r.status_code == 429 and wait_on_429 > 0:
                print(f"Error 429: Too many requests, Retrying")
                time.sleep(wait_on_429_time)
                wait_on_429 -= wait_on_429_time
                attempts += 1
            else: print(f"Request failed with error code: {r.status_code}, Retrying")
            return POST(key, parameters, attempts = attempts - 1 , timeout=timeout, wait_on_429=wait_on_429, wait_on_429_time=wait_on_429_time, minimum_delay = minimum_delay, wait_time_rand = wait_time_rand)
        if r.status_code == 200: print(f'API Request Completed in {time.time() - last_request_time}s')
        return r
    except requests.exceptions.Timeout as e:
        if attempts > 0: 
            print(f"Request Timed Out after {timeout} seconds, Retrying")
            return POST(key, parameters, attempts = attempts - 1 , timeout=timeout, wait_on_429=wait_on_429, wait_on_429_time=wait_on_429_time, minimum_delay = minimum_delay, wait_time_rand = wait_time_rand)
        return e
    except Exception as e:
        return e

def LOAD(response,parameters):
    if response is None or not hasattr(response,"status_code") or isinstance(response,Exception):
        if isinstance(response,requests.exceptions.Timeout):
            return None, 408
        return response, -1
        
    if response.status_code == 200:
        with ZipFile(BytesIO(response.content)) as zip_file:
            file_list = zip_file.namelist()
            image_file_name = file_list[0]
            image_data = zip_file.read(image_file_name)
            image = Image.open(BytesIO(image_data))
            return image, 200
    else:
        print(f"NAI Image Load Failure - Error Code: {response.status_code}")
        return None, response.status_code

def GET(key, attempts = 5, timeout = 30):          
    try:
        r = requests.get('https://api.novelai.net/user/subscription',headers=get_headers(key), timeout= timeout)
        if attempts > 0 and r is not None and r.status_code!= 200 and r.status_code not in TERMINAL_ERRORS:
            print(f"Request failed with error code: {r.status_code}, Retrying")
            return GET(key, attempts = attempts - 1 , timeout=timeout)
        return r
    except requests.exceptions.Timeout as e:
        if attempts > 0: return GET(key, attempts = attempts - 1 , timeout=timeout)
        print(f"Request Timed Out after {timeout} seconds, Retrying")
        return e
    except Exception as e:
        return e

def subscription_status(key):
    if not key: return -1,False,0,0        
    response = GET(key)
    if response is not None and isinstance(response,requests.exceptions.RequestException):
        print("Subscription Status Check Timed Out, Try Again.")
        return 408,False,0,0
    if response is None or not hasattr(response,'status_code'):
        print ("Unknown Error", response)
        return 404,False,0,0
    if response.status_code==200:
        content = response.json()
        def max_unlimited():
            max = 0
            for l in content['perks']['unlimitedImageGenerationLimits']:
                if l['maxPrompts'] > 0 and l['resolution'] >= max: max = l['resolution']
            return max
        max = max_unlimited()
        active = content['active']        
        unlimited = active and max >= 1048576        
        points = content['trainingStepsLeft']['fixedTrainingStepsLeft']+content['trainingStepsLeft']['purchasedTrainingSteps']        
        return response.status_code, unlimited, points, max
    else: print (response.status_code)
    return response.status_code,False,0,0


def tryfloat(value, default = None):
    try:
        value = value.strip()
        return float(value)
    except Exception as e:
        #print(f"Invalid Float: {value}")
        return default
        
def prompt_has_weight(p):
    uo = '('
    ux = ')'
    e=':'
    eidx = None
    inparen=0
    for i in range(len(p)):
        c = p[i]
        if c in [uo]:
            if i>0 and p[i]=='\\': continue
            inparen+=1
        elif c == e:
            eidx = i
        elif c == ux and inparen>0:
            inparen-=1
            if eidx is None: continue
            weight = tryfloat(p[eidx+1:i],None)
            if weight is not None:return True
    return False
    
def prompt_is_nai(p):
    return "{" in p
    
def prompt_to_nai(p, parenthesis_only = False):
    chunks = []
    out = ""
    states = []
    start = 0
    state = None
    do = '['
    dx = ']'
    uo = '('
    ux = ')'
    e=':'
    
    
    def addtext(i, end,prefix = "", suffix = ""):
        nonlocal out
        nonlocal state
        nonlocal start
        if state is None:
            s = p[start:i]
        elif state[3] is not None:
            s = state[3] + p[state[0]:i]
        else: s = p[state[0]:i]
        
        s = f'{prefix}{s}{suffix}'
        #print (s)
        if len(states)>0:
            next = states.pop()
            next[3] += s
            next[0] = end
            state = next
        else:
            state = None        
            out+=s
            
        start = end
        
    def adjustments(v):
        if v==1: return 1
        if v<=0: return 25
        if v < 1: v = 1/v
        m = 1
        for i in range(0,25):
            dif = v - m
            m*=1.05
            if v < m and dif <= m - v: return i
        return 25
    
    idx = 0
    while idx < len(p) or (state is not None and idx < 1000 + len(p)):
        if idx < len(p): 
            i = idx
            c = p[i]
        else: 
            c = ux if state[1] == uo else dx
            i = len(p)
        idx+=1        
        if c not in [do,uo,dx,ux,e] or i>0 and p[i-1] == '\\': continue    
        if c in [do,uo]:
            if parenthesis_only and c == do: continue
            if state is None: addtext(i,i+1)
            else:     
                state[3] += p[state[0]:i] 
                state[0] = i+1
                states.append(state)
            state = [i+1,c,None,""]                
        elif state is None: continue        
        elif c == e: state[2] = i
        elif c == dx and not parenthesis_only: addtext(i,i+1,'[[',']]' )
        elif c == ux:
            if state[2] is not None:
                numstart = state[2]+1
                numend = i
                weight = tryfloat(p[state[2]+1:i],None)
                if weight is not None:
                    adj = adjustments(weight)
                    if abs(weight - 1) < 0.025: addtext(state[2],i+1)
                    elif weight < 1: addtext(state[2],i+1,'['*adj ,']'*adj )
                    else: addtext(state[2],i+1,'{'*adj,'}'*adj )
                    continue
            if parenthesis_only: addtext(i,i+1,'(',')')
            else: addtext(i,i+1,'{{','}}')
    if start<len(p): addtext(len(p),len(p))
    if not parenthesis_only: out = out.replace("\\(","(").replace("\\)",")")
    return out


def prompt_to_a1111(p):
    p = re.sub("\\\\\\\\","\\\\",p) #remove escaped slashes
    p = re.sub("(?<!\\\\)([\\(\\)])","\\\\\\1",p) #escape parenthesis
    
    do = '['
    dx = ']'
    uo = '{'
    ux = '}'
    
    stack = [] # type, seqIndex
    seq = [[0,None,len(p),0]] # pIndex,weightType,otherIndex,weightCount    
    def push(i,c):
        if seq[-1][1] is None:
            if seq[-1][0] == i: seq.pop() # Remove Zero Length Sequence
            else: seq[-1][2]=i # Set Sequence End Point
        if c in [dx,ux]:
            o = do if c == dx else uo
            if len(stack) > 0 and stack[-1][0] == o: # if close, check for matching open
                closed = stack.pop()[1]
                seq[closed][2] = len(seq)
                seq.append([i+1,c,closed,1])
            else: c = do if c == ux else uo # No matching open, invert 
        if c in [do,uo]: 
            stack.append([c,len(seq)]) # push weight 
            seq.append([i+1,c,i+1,1])
        if i+1 < len(p): seq.append([i+1,None,len(p),0]) # Begin text sequence
        
    for i in range(len(p)): # build text/weight sequence
        if p[i] not in [do,uo,dx,ux] or i>0 and p[i-1] == '\\': continue
        push(i,p[i])

    for op in reversed(stack): # close open weights
        o = op[0]
        push(len(p), dx if o == do else ux)
    
    for i in range(1,len(seq)): # Combine adjacent weights
        c = seq[i][1]
        if c in [dx,ux] and c == seq[i-1][1] and seq[i-1][3] > 0:
            if seq[i][2] +1 == seq[i-1][2]:
                seq[i][3]+=seq[i-1][3]
                seq[seq[i][2]][3] = seq[i][3]
                seq[i-1][3] = 0
                seq[seq[i-1][2]][3] = 0
            
    for i in range(1,len(seq)): # remove empty weights
        c = seq[i][1]
        if c in [dx,ux] and seq[i][3]>0:
            content = False
            for x in range(seq[i][2]+1,i):
                if seq[x][1] is None and seq[x][0] < seq[x][2]:
                    content=True
                    break
            if not content:
                seq[i][3] = 0
                seq[seq[i][2]][3]=0
        
    def weight(up, cnt):
        weight = 1
        for i in range(cnt):
            if up: weight *= 1.05
            else: weight /= 1.05
        return weight
    
    out = ""
    for s in seq:
        c = s[1]
        o = s[0]
        x = s[2]
        if c is None and x - o > 0: out+=p[o:x]
        if c is not None and s[3] > 0:            
            if c in [do,uo]: out+='('
            if c == dx: out+=f':{weight(False,s[3]):.5g})'
            if c == ux: out+=f':{weight(True,s[3]):.5g})'

    return out
    
    
def NAIGenParams(prompt, neg, seed, width, height, scale, sampler, steps, noise_schedule, dynamic_thresholding= False, sm= False, sm_dyn= False, cfg_rescale=0,uncond_scale =1,model =NAIv3 ,image = None, noise=None, strength=None ,extra_noise_seed=None, mask = None,qualityToggle=False,ucPreset = 2,overlay = False,legacy_v3_extend = False,reference_image = None, reference_information_extracted = 1.0 , reference_strength = 0.6):
    def clean(p):
        if type(p) != str: p=f'{p}'
        #TODO: Look for a better way to do this        
        p=re.sub("(?<=[^\\\\])\"","\\\"" ,p)
        p=re.sub("\r?\n"," " ,p)
       ## p=re.sub("\s"," " ,p)
        return p
    prompt=clean(prompt)
    neg=clean(neg)
    
    if type(uncond_scale) != float and type (uncond_scale) != int: uncond_scale = 1.0
    if type(cfg_rescale) != float and type (cfg_rescale) != int: cfg_rescale = 0.0
    
    if prompt == "": prompt = " "    
    if model not in nai_models: model = NAIv3
    isV3 = model == NAIv3 or model == NAIv3f
    if "ddim" in sampler.lower():
        sampler = "ddim_v3" if isV3 else "ddim"
    elif sampler.lower() not in NAI_SAMPLERS: sampler = "k_euler"
    
    if "ddim" in sampler.lower() or not isV3: 
        noise_schedule=""
    else:
        if noise_schedule.lower() not in noise_schedules: 
            if sampler != "k_dpmpp_2m": noise_schedule = "native" 
            else:  noise_schedule = "exponential"
        if "_ancestral" in sampler and noise_schedule == "karras": noise_schedule = "native"
        noise_schedule = f',"noise_schedule":"{noise_schedule}"'
    
    cfg_rescale = f',"cfg_rescale":{cfg_rescale}' if isV3 else ""
    uncond_scale = f',"uncond_scale":{uncond_scale}' if isV3 or model == NAIv2 else ""
        
    if qualityToggle:
        if isV3:
            tags = 'best quality, amazing quality, very aesthetic, absurdres'
            if tags not in prompt: prompt = f'{prompt}, {tags}'
        elif model == NAIv2:
            tags = 'very aesthetic, best quality, absurdres'
            if not prompt.startswith(tags):
                prompt = f'{tags}, {prompt}'
        else:
            tags = 'masterpiece, best quality'
            if not prompt.startswith(tags):
                prompt = f'{tags}, {prompt}'    
    if ucPreset == 0:
        if isV3:
            tags = 'lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]'
        elif model == NAIv2:
            tags = 'lowres, bad, text, error, missing, extra, fewer, cropped, jpeg artifacts, worst quality, bad quality, watermark, displeasing, unfinished, chromatic aberration, scan, scan artifacts'
        else:
            tags = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
        if tags not in neg: neg = f'{tags}, {neg}'
    
    if ucPreset == 1:
        if isV3 or model == NAIv2:
            tags = 'lowres, jpeg artifacts, worst quality, watermark, blurry, very displeasing'
        else:
            tags = 'lowres, text, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
        if tags not in neg: neg = f'{tags}, {neg}'
        
    if ucPreset == 2:
        if isV3:
            tags = 'lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract], bad anatomy, bad hands, @_@, mismatched pupils, heart-shaped pupils, glowing eyes'
        elif model == NAIv2:
            tags = 'lowres, bad, text, error, missing, extra, fewer, cropped, jpeg artifacts, worst quality, bad quality, watermark, displeasing, unfinished, chromatic aberration, scan, scan artifacts'
            ucPreset = 0
        else:
            tags = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
            ucPreset = 0
        if tags not in neg: neg = f'{tags}, {neg}'
    
    if ucPreset == 3 and not isV3:
        ucPreset = 2
            
    if isinstance(image, Image.Image):            
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='PNG')
        image = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
    
    if image is not None:    
        action = 'img2img'
        image = f',"image":"{image}"'
        strength = f',"strength":{strength or 0.5}'
        noise = f',"noise":{noise or 0}'
        extra_noise_seed = f',"extra_noise_seed":{extra_noise_seed or seed}'        
        if isinstance(mask, Image.Image):
            image_byte_array = BytesIO()
            mask.save(image_byte_array, format="PNG")
            mask = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
        if mask is not None:
            if model != NAIv2:
                model += "-inpainting"
            mask = f',"mask":"{mask}"'
            
            action="infill"
            sm=False
            sm_dyn=False
            if "ddim" in sampler.lower():
                print("DDIM Not supported for Inpainting, switching to Euler")
                sampler = "k_euler"
        #else: overlay = False
    else:
        strength = ""
        noise=""
        image=""
        extra_noise_seed=""
        action = 'generate'
        
    dynamic_thresholding = "true" if dynamic_thresholding else "false"
    sm = "true" if sm else "false"
    sm_dyn = "true" if sm_dyn else "false"
    qualityToggle = "true" if qualityToggle else "false"
    legacy_v3_extend = "true" if legacy_v3_extend else "false"
    overlay = "true" if overlay else "false"

    reference = None    
    if reference_image is not None:
        if isinstance(reference_image, Image.Image):            
            image_byte_array = BytesIO()
            reference_image.save(image_byte_array, format='PNG')
            reference_image = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")        
            reference = f',"reference_image":"{reference_image}","reference_information_extracted":{reference_information_extracted or 1},"reference_strength":{reference_strength or 0.6}'
        elif isinstance(reference_image,list):
            imgs=None
            rextracts=None
            rstrengths= None
            for i in range(len(reference_image)):
                img = reference_image[i]
                if isinstance(img, Image.Image):
                    image_byte_array = BytesIO()
                    img.save(image_byte_array, format='PNG')
                    img = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
                    rextract = reference_information_extracted[i] if isinstance(reference_information_extracted,list) and len(reference_information_extracted) > i else (reference_information_extracted or 1.0)                    
                    rstrength = reference_strength[i] if isinstance(reference_strength,list) and len(reference_strength) > i else (reference_strength or 0.6)

                    imgs = f'"{img}"' if imgs is None else f'{imgs},"{img}"'
                    rextracts = f'{rextract}' if rextracts is None else f'{rextracts},{rextract}'
                    rstrengths = f'{rstrength}' if rstrengths is None else f'{rstrengths},{rstrength}'
            reference = f',"reference_image_multiple":[{imgs}],"reference_information_extracted_multiple":[{rextracts}],"reference_strength_multiple":[{rstrengths}]'
    
    return f'{{"input":"{prompt}","model":"{model}","action":"{action}","parameters":{{"params_version":1,"width":{int(width)},"height":{int(height)},"scale":{scale},"sampler":"{sampler}","steps":{steps},"seed":{int(seed)},"n_samples":1{strength or ""}{noise or ""},"ucPreset":{ucPreset},"qualityToggle":{qualityToggle},"sm":{sm},"sm_dyn":{sm_dyn},"dynamic_thresholding":{dynamic_thresholding},"controlnet_strength":1,"legacy":false,"legacy_v3_extend":{legacy_v3_extend},"add_original_image":{overlay}{uncond_scale or ""}{cfg_rescale or ""}{noise_schedule or ""}{image or ""}{mask or ""}{reference or ""}{extra_noise_seed or ""},"negative_prompt":"{neg}"}}}}'

def noise_schedule_selected(sampler,noise_schedule):
    noise_schedule=noise_schedule.lower()
    sampler=sampler.lower()
    
    if noise_schedule not in noise_schedule or sampler == "ddim": return False
    
    if sampler == "k_dpmpp_2m": return noise_schedule != "exponential"                 
    return noise_schedule != "native" 

def get_set_noise_schedule(sampler,noise_schedule):
    if sampler == "ddim": return ""
    if noise_schedule_selected(sampler, noise_schedule): return noise_schedule
    return noise_schedule_selections[0]
    
    
