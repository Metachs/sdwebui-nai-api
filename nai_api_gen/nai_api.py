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
import random
import json

NAIv1 = "nai-diffusion"
NAIv1c = "safe-diffusion"
NAIv1f = "nai-diffusion-furry"
NAIv2 = "nai-diffusion-2"
NAIv3 = "nai-diffusion-3"
NAIv3f = "nai-diffusion-furry-3"
NAIv4cp = "nai-diffusion-4-curated-preview"
NAIv4f = "nai-diffusion-4-full"
NAIv4 = "nai-diffusion-4-full"

NAI_IMAGE_URL = 'https://image.novelai.net/ai/generate-image'
NAI_AUGMENT_URL = 'https://image.novelai.net/ai/augment-image'

nai_models = [NAIv4f,NAIv4cp,NAIv3,NAIv3f,NAIv2]

NAI_SAMPLERS = ["k_euler","k_euler_ancestral","k_dpmpp_2s_ancestral","k_dpmpp_2m","ddim","k_dpmpp_sde","k_dpmpp_2m_sde"]
noise_schedules = ["exponential","polyexponential","karras","native"]
augment_modes = ["colorize","emotion","lineart","sketch","declutter","bg-removal"]
augment_emotions = ['neutral', 'happy', 'sad', 'angry', 'scared', 'surprised', 'tired', 'excited', 'nervous', 'thinking', 'confused', 'shy', 'disgusted', 'smug', 'bored', 'laughing', 'irritated', 'aroused', 'embarrassed', 'worried', 'love', 'determined', 'hurt', 'playful']

noise_schedule_selections = ["recommended","exponential","polyexponential","karras","native"]

nai_text_tag = '. Text:'

def get_headers(key):
    return {
        'accept': "*/*",
        'accept-language': "en-US,en;q=0.9,ja;q=0.8",   
        'Authorization':"Bearer "+ key,     
        'content-type': "application/json",
    }

TERMINAL_ERRORS = [401,403,404,-1]
TERMINAL_ERROR_MESSAGES = ["Error decoding request:","Error verifying request:"]

#TODO: Move to js file
vibe_processing_js ="""
async function (source){
    var img = new Image;
    img.src = source;
    await img.decode();

	var size = 448;
	var canvas = document.createElement("canvas");
	canvas.width = size;
	canvas.height = size;

	var ctx = canvas.getContext("2d");
	ctx.fillRect(0,0,canvas.width, canvas.height);
    
	width = img.width > img.height ? size : img.width * (size / img.height) ;
	height = img.height > img.width ? size : img.height * (size / img.width) ;
	ctx.drawImage(img, (size - width) / 2, (size - height) / 2, width, height) ;

	return canvas.toDataURL("image/png");
}
"""

def get_timeout(timeout, width, height, steps):

    histep = steps > 35
    
    midres = width * height > 768*768
    hires = width * height > 1280*1280
    
    if timeout > 60 : return timeout
    
    if histep and hires: return 95
    if hires: return 75
    
    if timeout > 30 : return timeout
    
    if midres and histep: return 45
    if midres or histep: return 35
    
    return 30
    
def POST(key,parameters, attempts = 0, timeout = 120, wait_on_429 = 0, wait_on_429_time = 5, attempt_count = 0,url = 'https://image.novelai.net/ai/generate-image'):
    try:
        r = requests.post(url,headers=get_headers(key), data=parameters.encode(),timeout= timeout) 
    except Exception as e:
        r = e
    return CHECK(r, key, parameters, attempts = attempts, timeout=timeout, wait_on_429=wait_on_429, wait_on_429_time=wait_on_429_time, attempt_count=attempt_count+1,url = url)

def CHECK(r,key,parameters, attempts = 0, timeout = 120, wait_on_429 = 0, wait_on_429_time = 5, attempt_count = 1,url = 'https://image.novelai.net/ai/generate-image'):
    r.message = ""
    r.files = []
    r.retry_count = attempt_count - 1
    if isinstance(r,requests.exceptions.Timeout):
        r.status_code = 408
        r.message = f"Request Timeout."        
        if attempt_count > attempts: return r
        print(f"Request Timed Out after {timeout} seconds, Retrying")
        return POST(key, parameters, attempts = attempts, timeout=timeout, wait_on_429=wait_on_429, wait_on_429_time=wait_on_429_time, attempt_count=attempt_count,url = url)
    elif not hasattr(r,'status_code'): 
        r.status_code = -1
        r.message = 'Unhandled Exception'
        return r
    elif r.status_code != 200:
        try: r.message = f"{r.status_code} Error: {json.loads(r.text).get('message')}" 
        except: r.message = f"{r.status_code} Error"
        print(r.message)
        if r.status_code in TERMINAL_ERRORS or any(tem in r.message for tem in TERMINAL_ERROR_MESSAGES): 
            return r
        if r.status_code == 429 and wait_on_429 > 0:
            print(f"Error 429: {r.message}, waiting {wait_on_429}s and retrying.")
            time.sleep(wait_on_429_time)
            wait_on_429 -= wait_on_429_time
            if attempt_count > 1: attempt_count -= 1 # Concurrent Request Failure only counts as a single failure
        elif attempt_count > attempts: return r
        else: 
            print(f"Error {r.status_code}: {r.message}, Retrying")
            time.sleep(attempt_count*2-1)
        return POST(key, parameters, attempts = attempts, timeout=timeout, wait_on_429=wait_on_429, wait_on_429_time=wait_on_429_time, attempt_count=attempt_count,url = url)
    else:
        try:
            with ZipFile(BytesIO(r.content)) as zip_file:
                for fn in zip_file.namelist():
                    r.files.append(Image.open(BytesIO(zip_file.read(fn))))
                return r
        except Exception as e:
            print(f"NAI Image Load Failure - Could not read image File")
            e.status_code = -1
            return e
    
def GPOST( params , attempts = 0, timeout = 120, wait_on_429 = 0, wait_on_429_time = 5,url = 'https://image.novelai.net/ai/generate-image'):
    import grequests
    rs=[]
    for key, parameters in params:
        rs.append(grequests.post('https://image.novelai.net/ai/generate-image',headers=get_headers(key), data=parameters.encode(),timeout= timeout))        
    rs = grequests.map(rs, exception_handler = lambda r,e:e)
    
    for ri in range(len(rs)):
        key,parameters = params[ri]
        rs[ri] = CHECK(rs[ri],key, parameters, attempts = attempts, timeout=timeout, wait_on_429=wait_on_429, wait_on_429_time=wait_on_429_time,url = url)
    return rs

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
        if isinstance(value,str): value = value.strip()
        return float(value)
    except Exception as e:
        #print(f"Invalid Float: {value}")
        return default
        
        
def get_vibe_encoding(key,model,reference_image,params,timeout = 120):
    if isinstance(reference_image, Image.Image):            
        image_byte_array = BytesIO()
        reference_image.save(image_byte_array, format='PNG')
        reference_image = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
    payload = {'image':reference_image, 'model' : model}
    for k in params.keys(): payload[k] = params[k]
    url = 'https://image.novelai.net/ai/encode-vibe'
    try:
        response = requests.post(url,headers=get_headers(key), data=json.dumps(payload).encode(),timeout= timeout) 
        if response.status_code==200: 
            return base64.b64encode(response.content).decode("utf-8")  
        try: response.message = f"{response.status_code} Error: {json.loads(response.text).get('message')}"
        except: response.message = f"{response.status_code} Error"
        print(response.message)
    except requests.exceptions.RequestException as e:
        print("encode-vibe Timed Out, Try Again Later.")    
    except Exception as e:
        print(f"Could not retrieve Encoding")
        print (e)        
    return None
    
vibe_model_names = {
    NAIv4f:"v4full", 
    NAIv4cp:"v4curated",
}

def create_encoding_file(encoding, model, name = None):
    hash = hashlib.sha256(encoding.encode()).hexdigest()
    if not name: name = f"{hash[:6]}-{hash[-6:]}"
    vmodel = vibe_model_names.get(model, 'v4full')
    file = {
        "identifier": "novelai-vibe-transfer",
        "version": 1,
        "type": "encoding",
        "id": hash,
        "encodings": {
            vmodel: {
                "unknown": {
                    "encoding": encoding,
                    "params": {}
                }
            }
        },
        "name": name,
        "createdAt": int(time.time()),
    }   
    return file


def create_vibe_bundle(vibes):
    file = {
        "identifier": "novelai-vibe-transfer-bundle",
        "version": 1,
        "vibes": vibes or [],
    }
    return file

def create_vibe_file(reference_image, name = None):
    image = None
    if isinstance(reference_image, Image.Image):
        image = reference_image
        image_byte_array = BytesIO()
        reference_image.save(image_byte_array, format='PNG')
        # TODO: Convert Image to known supported format
        reference_image = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
    hash = hashlib.sha256(reference_image.encode()).hexdigest()
    if image is None:
        image = Image.open(BytesIO(base64.b64decode(reference_image)))
    thumb = ""
    if isinstance(image, Image.Image):        
        image_byte_array = BytesIO()
        image.convert('RGB').resize(( int(image.width // (max(image.size)/256)), int(image.height // (max(image.size)/256)))).save(image_byte_array, format='JPEG')
        thumb = f'data:image/jpeg;base64,{base64.b64encode(image_byte_array.getvalue()).decode("utf-8")}'
        
    if not name: name = f"{hash[:6]}-{hash[-6:]}"
    file = {
        "identifier": "novelai-vibe-transfer",
        "version": 1,
        "type": "image",
        "image": reference_image,
        "id": hash,
        "encodings": { },
        "name": name,
        "thumbnail": thumb,
        "createdAt": int(time.time()),
        # "importInfo": {
            # "model": model,
            # "information_extracted": information_extracted,
            # "strength": strength
        # }
    }
    return file
    
    
    
def get_vibe_preview(vibe_file):
    text = None
    if vibe_file.get('type', None)=='encoding':
        return Image.new("RGBA",(256,256), color = "black")
    
    if 'thumbnail' in vibe_file: 
        text = vibe_file['thumbnail']
        if ',' in text:
            text = text.split(',',1)[1]
    if not text and 'image' in vibe_file:
        text = vibe_file['image']
    
    if text:
        try:
            return Image.open(BytesIO(base64.b64decode(text.encode())))
        except Exception as e:
            print("Error Loading Preview")
            print(e)
        
    return None

def has_encoding(vibe_file,information_extracted,model):
    if model not in vibe_model_names:
        print("Unknown Model, could not add encoding")
        return False
    vmodel = vibe_model_names[model]
    encodings =vibe_file['encodings'].get(vmodel,{})
    params = {'information_extracted':information_extracted}     
    paramkey= f'information_extracted:{information_extracted:.4f}'.rstrip('0').rstrip('.') #Python is FUCKING RETARDED    
    paramkey = hashlib.sha256(paramkey.encode()).hexdigest()
    if paramkey in encodings: return True
    return False
    
def get_encoding(vibe_file,information_extracted,model):
    if model not in vibe_model_names:
        print("Unknown Model, could not add encoding")
        return False
    vmodel = vibe_model_names[model]
    encodings =vibe_file['encodings'].get(vmodel,{})
    params = {'information_extracted':information_extracted}     
    paramkey= f'information_extracted:{information_extracted:.4f}'.rstrip('0').rstrip('.') #Python is FUCKING RETARDED    
    paramkey = hashlib.sha256(paramkey.encode()).hexdigest()
    return encodings.get(paramkey, {}).get('encoding', None)
    
def get_closest_ie(vibe_file,information_extracted,model, default = 1.0):
    if model not in vibe_model_names:
        print("Unknown Model, could not add encoding")
        return False
    vmodel = vibe_model_names[model]
    encodings =vibe_file['encodings'].get(vmodel,{})
    
    close = None
    
    for v in encodings.values():
        ie = tryfloat(v.get('params',{}).get('information_extracted',None))
        if close is None or ie is not None and abs(ie - information_extracted) < abs(close - information_extracted):
            close = ie
                
    return close or default
    
def get_closest_encoding(vibe_file,information_extracted,model):
    if model not in vibe_model_names:
        print("Unknown Model, could not add encoding")
        return False
    vmodel = vibe_model_names[model]
    encodings =vibe_file['encodings'].get(vmodel,{})
    
    close = None
    best = None
    
    for v in encodings.values():
        ie = tryfloat(v.get('params',{}).get('information_extracted',None))
        if close is None or ie is not None and abs(ie - information_extracted) < abs(close - information_extracted):
            close = ie
            best = v
                
    return best.get('encoding', None) if best is not None else None, close

def get_vibe_ie(vibe_file, model, defaultvalue = 1.0):
    if model not in vibe_model_names:
        print("Unknown Model, could not add encoding")
        return False
    vmodel = vibe_model_names[model]
    encodings =vibe_file['encodings'].get(vmodel,{})
    if encodings: 
        for v in encodings.values():
            i = tryfloat(v.get('params',{}).get('information_extracted',None))
            if i is not None: return i
    return defaultvalue
    
def get_vibe_presets(vibe, model):
    info = vibe.get('importInfo', None)
    if not info: return get_vibe_ie(vibe, model),0.6
    str = info.get('strength',0.6)
    ie = info.get('information_extracted', None)
    if ie is None or info.get('model', model) != model: return 1.0, str
    return ie, str

def add_encoding(key, vibe_file, information_extracted = 1.0, model = "nai-diffusion-4-full"):
    if model not in vibe_model_names:
        print("Unknown Model, could not add encoding")
        return False
    vmodel = vibe_model_names[model]
    encodings = vibe_file['encodings']
    params = {'information_extracted':information_extracted}     
    paramkey= f'information_extracted:{information_extracted:.4f}'.rstrip('0').rstrip('.') #Python is FUCKING RETARDED    
    paramkey = hashlib.sha256(paramkey.encode()).hexdigest()
    
    if vmodel in encodings:
        if paramkey in encodings[vmodel]:
            print("Encoding already Exists")
            return False
    else:
        encodings[vmodel] = {}   
    
    encoding = get_vibe_encoding(key,model,vibe_file['image'],params)
    if encoding is None: 
        print ("No Encoding!")
        return False
    encodings[vmodel][paramkey] = {'encoding':encoding, 'params':params }
    return True

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
    
def prompt_to_nai_v4(p, parenthesis_only = False):
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
                    addtext(state[2],i+1, f'{weight:.5g}::','::' )
                    
                    # if abs(weight - 1) < 0.025: addtext(state[2],i+1)
                    # elif weight < 1: addtext(state[2],i+1,'['*adj ,']'*adj )
                    # else: addtext(state[2],i+1,'{'*adj,'}'*adj )

                    continue
            if parenthesis_only: addtext(i,i+1,'(',')')
            else: addtext(i,i+1,'{{','}}')
    if start<len(p): addtext(len(p),len(p))
    if not parenthesis_only: out = out.replace("\\(","(").replace("\\)",")")
    return out
    
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

def get_skip_cfg_above_sigma(width,height): 
    return 19 * math.pow( width * height / (832 * 1216) , 0.5)

def clean_prompt(p):
    if type(p) != str: p=f'{p}'
    return p
    
def AugmentParams(mode, image, width, height, prompt, defry, emotion, seed=-1):
    prompt=clean_prompt(prompt)
    if mode == 'emotion' and emotion and emotion in augment_emotions:
        prompt = f"{emotion.lower()};;{prompt}"
    if prompt:
        prompt = f',"prompt":{json.dumps(prompt)}'
    else: prompt = ''
    defry = f',"defry":{int(defry)}'
    
    if isinstance(image, Image.Image):
        # print(image.mode, image.width, image.height)
        # image.save(r"D:\AI\SDout\ImageWut_"+f"{time.time()}.png")
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='PNG')
        image = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
        
    if image: image = f',"image":"{image}"' 
    else: image = f',"image":""'

    if mode not in ['colorize', 'emotion' ]: 
        prompt = ''
        defry = ''
    seed = f',"seed":{int(seed)}' if seed != -1 else ''
    # with open(r"D:\AI\SDout\ara_"+f"{time.time()}.txt", "w", encoding="utf8") as file:
        # file.write(f'{{"req_type":"{mode}","width":{int(width)},"height":{int(height)}{defry or ""}{prompt or ""}{seed or ""}}}')
    return f'{{"req_type":"{mode}","width":{int(width)},"height":{int(height)}{image or ""}{defry or ""}{prompt or ""}{seed or ""}}}'

def NAIGenParams(prompt, neg, seed, width, height, scale, sampler, steps, noise_schedule, dynamic_thresholding= False, sm= False, sm_dyn= False, cfg_rescale=0,uncond_scale =1,model =NAIv3 ,image = None, noise=None, strength=None ,extra_noise_seed=None, mask = None,qualityToggle=False,ucPreset = 2,overlay = False,legacy_v3_extend = False,reference_image = None, reference_information_extracted = None , reference_strength = None,n_samples = 1,variety = False,skip_cfg_above_sigma = None,deliberate_euler_ancestral_bug=None,prefer_brownian=None, characterPrompts = None, text_tag = None, legacy_uc = False,normalize_reference_strength_multiple = False):

    params = {
        'params_version':3,
        'controlnet_strength':1,
        'legacy':False,        
    }
    
    payload = {'parameters': params}
    
    prompt=clean_prompt(prompt)
    neg=clean_prompt(neg)
    
    if type(uncond_scale) != float and type (uncond_scale) != int: uncond_scale = 1.0
    if type(cfg_rescale) != float and type (cfg_rescale) != int: cfg_rescale = 0.0
    if prompt == "": prompt = " "
    
    if model not in nai_models: model = NAIv3
    
    isV4 = model == NAIv4cp or model == NAIv4f
    isV3plus = model == NAIv3 or model == NAIv3f or isV4
    
    if isV4 and text_tag is None and nai_text_tag in prompt:
        # Documented Syntax (handle period before ' Text:')
        # Disabled for compatability, if user adds a period before ' Text:' NAI just ignores it.
        # if nai_text_tag in prompt: prompt, text_tag = prompt.split(nai_text_tag,1)
        
        # Actual Syntax (Only check for ' Text:')
        if ' Text:' in prompt: prompt, text_tag = prompt.split(' Text:',1)
    
    if "ddim" in sampler.lower():
        sampler = "ddim_v3" if isV3 else "ddim"
    elif sampler.lower() not in NAI_SAMPLERS: sampler = "k_euler"
    
    if "ddim" in sampler.lower() or not isV3plus: 
        noise_schedule=""
    else:
        if noise_schedule.lower() not in noise_schedules: 
            if sampler != "k_dpmpp_2m": noise_schedule = "native" 
            else: noise_schedule = "karras"
            
        if sampler == "k_euler_ancestral" and noise_schedule.lower() != "native":
            deliberate_euler_ancestral_bug = False if deliberate_euler_ancestral_bug is None else deliberate_euler_ancestral_bug
            prefer_brownian = True if prefer_brownian is None else prefer_brownian

        params['noise_schedule'] = noise_schedule
        params['deliberate_euler_ancestral_bug'] = bool(deliberate_euler_ancestral_bug)
        params['prefer_brownian'] = bool(prefer_brownian)  
    
    if isV3plus: params['cfg_rescale'] = float(cfg_rescale)
    
    skip_cfg_above_sigma = tryfloat(skip_cfg_above_sigma, 0) or None    
    params['skip_cfg_above_sigma'] = get_skip_cfg_above_sigma(width, height) if variety and not skip_cfg_above_sigma else skip_cfg_above_sigma
    
    if qualityToggle:
        if model == NAIv3:
            tags = 'best quality, amazing quality, very aesthetic, absurdres'
            if tags not in prompt: prompt = f'{prompt}, {tags}'
        elif model == NAIv4cp:
            tags = 'rating:general, best quality, very aesthetic, absurdres'
            if tags not in prompt: prompt = f'{prompt}, {tags}'
        elif isV4:
            if not text_tag: tags = 'no text, best quality, very aesthetic, absurdres'
            else: tags = 'best quality, very aesthetic, absurdres'
            if tags not in prompt: prompt = f'{prompt}, {tags}'
        elif model == NAIv3f:
            tags = '{best quality}, {amazing quality}'
            if tags not in prompt: prompt = f'{prompt}, {tags}'
        elif model == NAIv2:
            tags = 'very aesthetic, best quality, absurdres'
            if not prompt.startswith(tags):
                prompt = f'{tags}, {prompt}'
        else:
            tags = 'masterpiece, best quality'
            if not prompt.startswith(tags):
                prompt = f'{tags}, {prompt}'    
                
        
    tags=None
    if ucPreset == 2:
        if model == NAIv3:
            tags = 'lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract], bad anatomy, bad hands, @_@, mismatched pupils, heart-shaped pupils, glowing eyes'
        else:
            ucPreset = 0

    if ucPreset == 0:
        if model == NAIv3:
            tags = 'lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]'
        elif model == NAIv3f:
            tags = '{{worst quality}}, [displeasing], {unusual pupils}, guide lines, {{unfinished}}, {bad}, url, artist name, {{tall image}}, mosaic, {sketch page}, comic panel, impact (font), [dated], {logo}, ych, {what}, {where is your god now}, {distorted text}, repeated text, {floating head}, {1994}, {widescreen}, absolutely everyone, sequence, {compression artifacts}, hard translated, {cropped}, {commissioner name}, unknown text, high contrast'
        elif model == NAIv2:
            tags = 'lowres, bad, text, error, missing, extra, fewer, cropped, jpeg artifacts, worst quality, bad quality, watermark, displeasing, unfinished, chromatic aberration, scan, scan artifacts'
        elif model == NAIv4cp:
            tags = 'blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, logo, dated, signature, multiple views, gigantic breasts, white blank page, blank page'
        elif isV4:
            tags = 'blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, multiple views, logo, too many watermarks, white blank page, blank page'
        else:
            tags = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
    
    if ucPreset == 1:
        if model == NAIv3 or model == NAIv2:
            tags = 'lowres, jpeg artifacts, worst quality, watermark, blurry, very displeasing'
        elif model == NAIv4cp:
            tags = 'blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, logo, dated, signature, white blank page, blank page'
        elif isV4:
            tags = 'blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, white blank page, blank page'
        elif model == NAIv3f:
            tags = '{worst quality}, guide lines, unfinished, bad, url, tall image, widescreen, compression artifacts, unknown text'
        else:
            tags = 'lowres, text, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
    
    if tags and tags not in neg: neg = f'{tags}, {neg}'
    
    if isV4 and text_tag:
        prompt = f'{prompt}{nai_text_tag}{text_tag}'
        
    
    if ucPreset not in [0,1,2,3]: ucPreset = 3
    if ucPreset == 3 and model != NAIv3: ucPreset = 2

    if isinstance(image, Image.Image):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='PNG')
        image = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
    
    if image is not None:    
        action = 'img2img'
        params['image']=image
        params['strength'] = tryfloat(strength, 0.5)
        params['noise'] = tryfloat(noise, 0.0)
        params['extra_noise_seed'] = int(extra_noise_seed or seed)
        if isinstance(mask, Image.Image):
            image_byte_array = BytesIO()
            mask.save(image_byte_array, format="PNG")
            mask = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
        if mask is not None:
            if model != NAIv2: model += "-inpainting"
            params['mask']=mask
            action="infill"
            sm=False
            sm_dyn=False
            if "ddim" in sampler.lower():
                print("DDIM Not supported for Inpainting, switching to Euler")
                sampler = "k_euler"
        #else: overlay = False
    else:
        action = 'generate'
        
    if reference_image is not None:
        imgs=[]
        rextracts=[]
        rstrengths= []
        for i in range(len(reference_image)):
            img = reference_image[i]
            if isinstance(img, Image.Image):
                image_byte_array = BytesIO()
                img.save(image_byte_array, format='PNG')
                img = base64.b64encode(image_byte_array.getvalue()).decode("utf-8")
            if img:
                rextract = reference_information_extracted[i] if reference_information_extracted and len(reference_information_extracted) > i else 1.0
                rstrength = reference_strength[i] if reference_strength and len(reference_strength) > i else 0.6
                imgs.append(img)
                rextracts.append(tryfloat(rextract, 1.0))
                rstrengths.append(tryfloat(rstrength, 0.6))
        if normalize_reference_strength_multiple and sum(abs(n) for n in rstrengths)>1.0:
            total = sum(abs(n) for n in rstrengths)
            rstrengths = [f/total for f in rstrengths]
            
        if imgs is not None: 
            if isV4: params['normalize_reference_strength_multiple'] = bool(normalize_reference_strength_multiple)
            else: params['reference_information_extracted_multiple'] = rextracts
            params['reference_image_multiple']=imgs
            params['reference_strength_multiple']=rstrengths
    if isV4:
        if not characterPrompts or not isinstance(characterPrompts, list): characterPrompts = []
        use_coords = any(['center' in c for c in characterPrompts])
        cps = []
        ccp = []
        ccn = []
        v4p = {'caption':{'base_caption': prompt, 'char_captions': ccp}, 'use_coords':use_coords, 'use_order':True}
        v4n = {'caption':{'base_caption': neg, 'char_captions': ccn}, 'legacy_uc': bool(legacy_uc)}
        for cp in characterPrompts:
            if not isinstance(cp, dict): continue
            if 'center' not in cp: cp['center'] = {'x':0.5,'y':0.5}
            ccp.append({'char_caption': cp.get('prompt',""), 'centers':[cp['center']]})
            ccn.append({'char_caption': cp.get('uc',""), 'centers':[cp['center']]})
            cps.append(cp)
        params.update(characterPrompts= cps,v4_prompt= v4p,v4_negative_prompt=v4n,use_coords= bool(use_coords))
        params['legacy_uc'] = bool(legacy_uc)

    params['negative_prompt'] = neg
    params['legacy_v3_extend'] = legacy_v3_extend
    params['scale'] = float(scale)
    params['add_original_image'] = bool(overlay)
    params['dynamic_thresholding'] = bool(dynamic_thresholding)
    
    payload.update(input=prompt, action=action, model=model)    
    params.update(sampler=sampler, n_samples=n_samples, seed=seed, width=width, height=height, steps=steps, ucPreset=ucPreset, qualityToggle=qualityToggle)    
    
    if not isV4:
        params['sm'] = sm
        params['sm_dyn'] = sm_dyn
    else: 
        params['autoSmea'] = sm
    return json.dumps(payload)
    
def GrayLevels(image, inlo = 0, inhi = 255, mid = 128, outlo = 0, outhi = 255):
    from PIL import Image,ImageMath
    inlo,inhi,mid,outlo,outhi = tryfloat(inlo,0),tryfloat(inhi,255),tryfloat(mid,128),tryfloat(outlo,0),tryfloat(outhi,255)
    print(inlo,inhi,mid,outlo,outhi)
    image = image.convert("L")
    gamma = None
    if mid < 127:
        gamma = 1 / min(1 + ( 9 * ( 1 - mid/255 * 2 ) ), 9.99 )
    elif mid > 128:
        gamma = 1 / max( 1 - (( mid/255 * 2 ) - 1) , 0.01 )
    if inlo>0 or inhi < 255:
        image = ImageMath.eval( f'int((((float(x)/255)-{inlo/255})/({inhi/255}-{inlo/255}))*255)' ,x = image)
    if gamma is not None:
        image = ImageMath.eval( f'int((((float(x)/255) ** {gamma}))*255)' ,x = image)
    if outlo>0 or outhi < 255:
        image = ImageMath.eval( f'int(((float(x)/255)*{outhi/255-outlo/255}+{outlo/255})*255)' ,x = image)
    return image.convert("RGB")
