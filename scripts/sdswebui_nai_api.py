from modules import scripts
from nai_api_gen import nai_api_xyz,nai_api_processing,nai_api_settings,nai_stealth_text_info
from nai_api_gen.nai_api_script import NAIGENScriptBase

nai_api_xyz.script_setup()
nai_api_processing.script_setup()
nai_api_settings.script_setup()
nai_stealth_text_info.script_setup()

class NAIGENScript(NAIGENScriptBase):

    def __init__(self):
        super().__init__()    
        self.NAISCRIPTNAME = "NAI API Generation"    
        
    def show(self, is_img2img):
        return scripts.AlwaysVisible
