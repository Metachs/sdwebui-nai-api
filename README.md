# sdwebui-nai-api - Novel AI Image Gen in stable-diffusion webui

### An an extension for A1111's stable-diffusion-webui that pulls images using NovelAI's Image Generation Tool API.

#### Requires an active Novel AI account with Opus or Anlas to spend (NOT FREE). You will need to generate a NovelAI API Token for your account, and enter it into the NAI API section in sd-webui settings menu. See [https://docs.sillytavern.app/usage/api-connections/novelai] for details on generating an API key.

Install using "Extensions > Install from URL" using the repo's URL.

- Supports Text2Image, Image2Image, and Inpainting using NovelAI's API
- Allows feeding NAI generations directly into A1111's Local Image2Image, for a sort of multi-model hi-res fix.
- Allows use of a number of simple A1111 features/extensions, such as XYZ Grid generation. 
- Allows much higher precision Inpainting masks, as well as uploading masks from other sources.
- Automatically translates A1111 format weighted prompts to NovelAI's format eg (1girl:1.1) > {{1girl}} (Values can't be 1:1, but should be close enough)
- Includes a modified version of the Stealth PNG Info extension to fully support NovelAI's stealth PNG info, for sites that strip metadata.
- Enforce cost limits to avoid wasting Anlas when you have Opus

Most locally run extension are NOT supported, obviously. It does support basic extensions that only modify generation parameters, such as wildcards. Running this with certain extensions on may break things. 
 
### Note that this is compltely unaffiliated with NovelAI, and I can make no guarantees as to whether this is an allowable usage of Novel AI's services. However, since they provide the means for you to generate an API Token, I believe it is an acceptable use. IE use at you own risk, but you are probably ok if you don't do something stupid like leave generate forever on for days.


#### Unsupported/TODO
- NAI ControlNet Tools - Not sure how CN works in NAI, not worth worrying about until NAIv3 supports 'em
- NAI Batch support - This isn't free in the only NAI offering worth paying for, so I haven't bothered. (A1111 Batches are pulled sequentially)
- Separate Prompts for NAI/Local in Two pass Image2Image mode.
