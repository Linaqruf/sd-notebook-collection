# **Cagliostro Colab UI**
All-in-One, Customizable and Flexible AUTOMATIC1111's Stable Diffusion Web UI for Google Colab. <br>
 
## What's New?

### v.3.0.0 (10/06/23)
- Rewrote `cagliostro-colab-ui` codebase from scratch.
  - Used in-house module [Colablib](https://github.com/Linaqruf/colablib) as the primary library for go-to functions, such as colored print, git function, download syntax, etc.
  - Added numerous trivial but important pieces of information, such as the Python version, torch version, current commit hash.
  - Improved console logs, using `print_line()` and `cprint()` from `Colablib`.
- Merged `output_to_drive` with `mount_drive`.
  - Now, the output path is automatically set to drive if Google Drive is mounted.
- Built-in wildcard support.
- Renamed `dpm_v2_patch` to `dpmpp_2m_v2_patch`.
- Added new section: `Optimization config`.
  - Introduced `mobile_optimizations` to keep Colab tab alive for mobile users.
  - Removed `Keep Tab Alive for Mobile` due to similar functionality with the new optimization.
- New extensions!
  - [ilian6806/stable-diffusion-webui-state](https://github.com/ilian6806/stable-diffusion-webui-state) : Preserves Web UI parameters (inputs, sliders, checkboxes, etc.) after page reload.
  - [pkuliyi2015/multidiffusion-upscaler-for-automatic1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) : Offers tiled Diffusion and VAE optimization.
  - [Zuellni/Stable-Diffusion-WebUI-Image-Filters](https://github.com/Zuellni/Stable-Diffusion-WebUI-Image-Filters) : A simple image postprocessing extension using the Pillow library.
  - [Linaqruf/Umi-AI-debloat](https://github.com/Linaqruf/Umi-AI-debloat) : Wildcard manager, a fork of [Tsukreya/Umi-AI-debloat](https://github.com/Tsukreya/Umi-AI-debloat) that has already fixed the lowercase problem.
  - [AlUlkesh/sd_delete_button](https://github.com/AlUlkesh/sd_delete_button) : Adds a delete button for Automatic1111 txt2img and img2img.
- New default models!
  - Added [AnyLoRA_Anime_Mix](https://civitai.com/models/84586/) as a new AnyLoRA variant for 2D Anime generation, similar to NovelAI.
  - Renamed the old `AnyLoRA` to [AnyLoRA_Default](https://civitai.com/models/23900/).
  - Added [Ghost_Note_Delta](https://huggingface.co/corechan/GhostNotes#ghostnotedelta_m0528), [SDHK_V3](https://civitai.com/models/82813), for another Anime Model.
  - Added [Majic_Mix_V5]() to replace `Chillout Mix`.
  - Replaced `replicant_v2` with the latest and improved version, [Replicant_V3](https://huggingface.co/gsdf/Replicant-V3.0).
  - Reintroduced `Illuminati Diffusion V1.1`.
  - Removed Waifu Diffusion 1.5 models.
- New default VAEs!
  - Added [Blessed VAE](https://huggingface.co/NoCrypt/blessed_vae).
  - All default VAEs are now in `.safetensors`! Thanks to NoCrypt.
- ControlNet
  - Added new Annotator, up to `Lama cleaner`.
  - Included a list of new SDv2.x ControlNet Models from [thibaud/controlnet-sd21](https://huggingface.co/thibaud/controlnet-sd21).
  - Added `Custom ControlNet Model` section to download custom controlnet models such as [Illumination](https://huggingface.co/ioclab/control_v1u_sd15_illumination_webui), [Brightness](https://huggingface.co/ioclab/control_v1p_sd15_brightness), the upcoming [QR Code](https://www.reddit.com/r/StableDiffusion/comments/141hg9x/controlnet_for_qr_code/) model, and any other unofficial ControlNet Model.
  - Please ensure your custom ControlNet model has `sd15`/`sd21` in the filename.
  - Reintroduced `t2i_adapter_model`.
- Custom Download Corner
  - Added instructions for using the `fuse:` prefix, handling multiple URLs, and loading models from Google Drive.
  - The `prune:`, `fp16:`, or `fp32:` prefixes currently are not available.
  - Introduced a new section: `Download From Textfile`.
    - This uses a similar approach to the [etherealxx/batchlinks-webui](https://github.com/etherealxx/batchlinks-webui) extensions.
    - Provide a custom download URL for a `.txt` file instead of using the URL field. Edit the file: `/content/download_list.txt`.
    - Available hashtags: `#model`, `#vae`, `#embedding`, `#lora`, `#hypernetwork`, `#extensions`, `#upscaler`. Aliases are not currently supported.
    - Alternatively, you can input your `.txt` file into `custom_download_list_url`. This works for `pastebin`.

    Example: 
    ```python
    #model
    url1
    url2

    #lora
    url1 | filename1

    #embedding
    fuse:path
    ```
- Launch
  - Added `--opt-sdp-mem-attention` option for `accelerator`.
  - Included error handling if the user forgets to download the model and vae. It will automatically download these two files before launch:
    - Model: [AnyLoRA_Anime_Mix](https://civitai.com/models/84586/)
    - VAE: [Animevae](https://huggingface.co/NoCrypt/resources/blob/main/VAE/any.vae.safetensors)
  - Added `Token Merging Ratio` or `ToME SD` and `Negative Guidance Scale` to quicksettings
- Refactored `Download Generated Images` and other `Extras` cells by wrapping code inside functions.

- Bugfixes:
  - Fixed `dpmpp_2m_v2_patch`, now it's working properly.
  - Temporary fixed LoRA not applied when using Hires Fix
  - Fixed file extensions undefined in `Custom Download Corner`

### v.2.6.2 (17/05/23)
- Bugfixes:
  - Attempting to fix generate button stuck problem by adding `latest_gradio` option to force update `gradio` to `3.31`

### v.2.6.1 (17/05/23)
- Added `sd-civitai-browser` extension back, but this time using [SignalFlagZ/sd-civitai-browser](https://github.com/SignalFlagZ/sd-civitai-browser) version.
- Changed `stable-diffusion-webui-latent-two-shot` extension with [ashen-sensored](ashen-sensored/stable-diffusion-webui-two-shot)'s fork.
- Updated alternative tunnels recommendation: `ngrok` > `gradio` > `cloudflared` > `remotemoe` > `localhostrun` > `googleusercontent`. The ranking is based on the following reasons:
  + Gradio: Faster and slightly more stable than the others, but may have queue problems.
  - Cloudflared: Slow to start the application, loads extra networks and images.
  - Remotemoe: Currently not usable.
  - Localhostrun: Often gets disconnected, requiring a forced restart of cells to make it work.
  - Googleusercontent: -
- Bugfixes:
  - Fixed `stable-diffusion-webui-composable-lora` not being installed in `repo_type` values: `["AUTOMATIC1111", "AUTOMATIC1111-dev"]`.
  - Set `commit_hash` to empty and use latest commit because the bugs already fixed.


### v.2.6.0 (16/05/23)
- Updated to the latest commit.
- Available `repo_type` values: `["AUTOMATIC1111", "AUTOMATIC1111-dev", "Anapnoe"]`.
- Updated the new tile model with `control_v11f1e_sd15_tile_fp16.safetensors`.
- Added 2 optimization options:
  - `ram_alloc_patch`, also known as the Camenduru Patch, to decrease RAM usage.
  - `colab_optimizations`, also known as TheLastBen Patch, to load the model in VRAM and fix the gradio queue problem.
- Added the `use_presets` option to turn on or off the default prompt.
- Simplified the `Arguments` category options.
- Set a default value for `gradio_auth`.
- Changed [yfszzx/stable-diffusion-webui-images-browser](https://github.com/yfszzx/stable-diffusion-webui-images-browser) to [zanllp/sd-webui-infinite-image-browsing](https://github.com/zanllp/sd-webui-infinite-image-browsing).
- Added the extension [butaixianran/Stable-Diffusion-Webui-Civitai-Helper](https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper).
- Added the extension [Bing-su/adetailer](https://github.com/Bing-su/adetailer).
- Added the extension [canvas-zoom](https://github.com/richrobber2/canvas-zoom).
- Added negative embeddings [EasyNegativeV2](https://huggingface.co/gsdf/Counterfeit-V3.0/blob/main/embedding/EasyNegativeV2.safetensors).
- Added a new theme for `Anapnoe-webui`: `["minimal", "minimal-orange"]`.
- Bug Fixes:
  - Fixed Inpaint not showing images [#28](https://github.com/Linaqruf/sd-notebook-collection/issues/28). It looks like [Bing-su/adetailer](https://github.com/Bing-su/adetailer) and [canvas-zoom](https://github.com/richrobber2/canvas-zoom) are unusable in `Anapnoe-webui`.
  - Fixed the `quicksettings` problem after the master repo was migrated from strings to a list. [#27](https://github.com/Linaqruf/sd-notebook-collection/issues/27) [#29](https://github.com/Linaqruf/sd-notebook-collection/issues/29).
  - An annoying bug has recently occurred in `Anapnoe-webui`. Extra networks are not displaying Models, Embeddings, LoRA, etc., even if they exist in the respective folder. Temporary fixed by reset commit hash to [f2b9c2c](https://github.com/anapnoe/stable-diffusion-webui-ux/commit/f2b9c2cb4fa5f0e866c1b6f84e44d12ff6653af3). Track discussion [here](https://github.com/anapnoe/stable-diffusion-webui-ux/issues/141)

### v.2.5.3 (21/04/23)
- `AUTOMATIC1111's stable-diffusion-webui` has been removed because Google Colab has prohibited the usage of any string named `stable-diffusion-webui` due to its massive usage of webui.
- `Anapnoe UI` has been set as default forcibly, also added warning messages.
- An `experimental` section has been added to install the `Anapnoe UI` integrated with `gradio 3.23.0`.
- `sd-webui-tunnels` forked by `camenduru` has been removed since the new ToS of Colab prohibits SSH shell or any similar tools.
- Now, there are two versions of the notebook available: 
   - [Cagliostro Colab UI](https://colab.research.google.com/github/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb): A compact and lightweight version that removes `AUTOMATIC1111's stable-diffusion-webui`.
   - [Cagliostro Colab UI Pro](https://colab.research.google.com/github/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui-pro.ipynb): It has all the features of the Cagliostro Colab UI update in 19/04. However, since I do not have a Colab Pro subscription, I am unlikely to maintain it.

  - 
### v.2.5.2 (19/04/23)
- Reformat and simplified most cells
- Rewording all available variable to make it easier to read, example: `git_pull` to `update_webui`
- Added `output_drive_folder` to customize gdrive outputs folder name
- Removed `clean_install` option
- Merged `load_v2_in_vram` and `merge_in_vram` to `colab_optimization` and set to `False`
- Deleted `stable_diffusion_v_1_5` and `replicant_v1` from available models
- Added `custom_upscaler_url` back

### v.2.5.1 (18/04/23)
- Added `cldm_config.yaml` for every ControlNet model (refer to [link](https://github.com/Mikubill/sd-webui-controlnet#download-models))
- Removed `custom_control_url` from `Custom Download Corner` cell
- Removed `illuminati_diffusion_v1_1` from `Available SDv2.x Model` list as the model has become exclusive to a certain image generation website
- Reworded certain arguments to make them more readable
- Removed `medvram` option from the `Arguments` list.

 ### v.2.5.0 (16/04/23)
- Update Web UI to the latest version
- Moved `What's new?` to [GitHub repository](https://github.com/Linaqruf/sd-notebook-collection/blob/main/README.md)
- Added link to [Cagliostro Colab UI User Manual](https://github.com/Linaqruf/sd-notebook-collection/blob/main/MANUAL.md)
- Added `cheatsheet` in every header and subheader based on [Cagliostro Colab UI User Manual](https://github.com/Linaqruf/sd-notebook-collection/blob/main/MANUAL.md) explanation
- Reformatted Notebook to be more readable.
- If `use_anapnoe_ui` set to `True`, skip updating `stable-diffusion-webui-images-browser` when `update_extensions` set to `True`
- Added 2 more `os.environ` changes
  - os.environ["PYTHONDONTWRITEBYTECODE"]='1'
  - os.environ['PYTHONWARNINGS'] = 'ignore'
- Added [Replicant V2.0](https://huggingface.co/gsdf/Replicant-V2.0) as new SDv2.x model
- [**ControlNet V1.1**](https://github.com/lllyasviel/ControlNet-v1-1-nightly) Update!
  - Removed old Annotator and ControlNet V1.0 model, and added new ones. Total: 13 new Annotator and 14 new ControlNet V1.1 models.
  - Renamed `sd21_control_model` to `sd21_control_v10_sd21_model`
  - Renamed `wd15_control_model` to `wd15_control_v10_sd21_model`
- Revamped how `Custom Download Corner` works.
  - Removed `custom_upscaler_url`, `custom_control_url`, and `custom_components_url`
  - Added a feature to prune model, fuse folder, copy from Google Drive. Users can find how to use it in this [link](https://github.com/Linaqruf/sd-notebook-collection/blob/main/MANUAL.md#custom-download-corner) or this table.

| Feature | Description | How to Use | Example |
| --- | --- | --- | --- |
| **Multiple Downloads** | Download multiple files at once. | Fill in the URL fields with the links to the files you want to download. Separate multiple URLs with a comma. | `url1, url2, url3` |
| **Auto-prune** | Prune models after downloading | Add `fp16:` or `fp32:` before URLs. | `fp16:url1` |
| **Copy from Google Drive** | Copy models from Google Drive and load them in the session. | Make sure you have already mounted Google Drive. Type the path to your model/lora/embedding from Google Drive. | `/content/drive/MyDrive/path/to/folder` |
| **Fusing Folder** | Fuse models/embeddings/LoRA folder to `/content/fused/{category}`. | Make sure you have already mounted Google Drive. Add `fuse:` before the path to the folder. | `fuse:/path/to/gdrive/folder` |
| **Auto-extract** | Extract files after downloading. | Add `links/to/file` ending with `.zip` or `.tar.lz4`. Extract files to specified destination directory. | `https//link.com/to/file` |
| **Install Extensions** | Install extensions for Stable Diffusion Web UI. | Add the link to the GitHub repository to `custom_extension_url`. | `https://github.com/user/repo` |

- Changed the Gradio authentication logic to use a `boolean` instead. The username is set to `cagliostro`, and the password is a randomly generated 6-character combination of ASCII letters and numbers.
- Added an option to enable `--opt-sdp-attention` instead of `xformers` to accelerate PyTorch 2.0.

 ### v.2.2.0 (03/04/23)
  - Update Web UI to the latest version
  - Update xformers to `0.0.18`
  - `git checkout` for extension using `Gradio 3.23` or above
  - If using Anapnoe UI/UX, users can't `update_extensions` at the moment to prevent code conflict because of different gradio version

 ### v.2.1.0 (31/03/23)
  - Added Reinstall logic
  - Fix bugs when upload images to ControlNet canvas, the images stretched
  - Fix bugs when Web UI can't starting if `use_anapnoe_ui` set to **True**
  - Changed `use_anapnoe_ui` logic by downloading and unpacking separate repo instead of `git remote origin set-url` and `git reset --hard`
  - Added **Theme Selector** section to changed default (anapnoe-webui) theme before starting Web UI

### v.2.0.1 (27/03/23)
  - Clean install and update Web UI to latest version, commit hash : `955df7751eef11bb7697e2d77f6b8a6226b21e13`
  - Manually edit `config.json` and `ui-config.json`
  - Change `auto_model` and `auto_vae` logic, change `config.json` instead of using `--vae_path` and `--ckpt`

### v.2.0.0 (27/03/23)
  - Reformat notebook with [black python formatter](https://github.com/psf/black)
  - Update Web UI and all extensions version (not latest)
  - Downgrade `xformers` to `0.0.16`
  - Downgrade `triton` to `2.0.0`
  - Set both `lora_dir` and `additional_networks_extra_lora_path` path to `os.path.join(repo_dir, "models/Lora")`
  - Set `lora_dir` path to `os.path.join(repo_dir, "models/ControlNet")`
  - Reset Anapnoe UI commit hash to `802fb8a16ba4bdbfba0dca55f9cdb265f4bd86f2`
  - Force reset commit hash to `3b47b000199ea8baf724080936ef985f53b3d081` when `use_anapnoe_ui` is enabled
  - Added `mount_drive` function
  - Added more error handling
  - Set some environment variables:
    - os.environ['colab_url'] = eval_js("google.colab.kernel.proxyPort(7860, {'cache': false})")
    - os.environ["LD_PRELOAD"] = "libtcmalloc.so"
    - os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    - os.environ["SAFETENSORS_FAST_GPU"]= '1'
  - Added 3 new default model
    - [Anything V3.0](https://huggingface.co/Linaqruf/anything-v3.0)
    - [AnyLoRA](https://huggingface.co/Lykon/AnyLoRA)
    - [Anime Pastel Dream](https://huggingface.co/Lykon/AnimePastelDream)
  - Deleted 3 default model
  - Anything V3.2
  - Anything V3.3
  - HoloKuki V2
- Changed `Chillout-Mix` to `Chillout-Mix-Ni`
- Changed `Illuminati V1.1` download link from CivitAI to Huggingface
- Put all ControlNet things to new separated cell
  - Added `pre_download_annotator` back
  - Added `sd21_control_model`
  - Added `wd15_control_model`
  - Added option to change ControlNet model and Adapter config outside Web UI
- Added support for load custom url from Google Drive, tricky, by copying from Google Drive to destination path.
- Set `--multiple` as default alternative tunnels, please don't use Gradio URL at the moment
  - Recommended Tunnels:
  - `ngrok` > `cloudflared` > `remotemoe` > `localhostrun` > `googleusercontent` > `gradio`

### v.1.4.1 (11/03/23)
- Trying to establish my own style, deleted all possible NoCrypt and TheLastBen's code legacy
- Temporary removed `pre_download_annotator`
- Deleted `PastelMix` from available models because the model license sold to [fantasy.ai](https://fantasy.ai/)
- Deleted `RefSlave-V2` from available models because the author removed his model from huggingface and civitai
- Added HoloKuki V2 as to available models
- Update t2i adapter model, currently has 8 model

### v.1.4 (09/03/23)
- Update xformers pre-compiled wheels to `xFormers 0.0.17.dev466`
- Update pre-installed dependencies for `Python 3.9.16`
- Added new Stable Diffusion Web UI Extensions: `batchlinks-webui` and `sd-webui-llul`

### v.1.3 (01/03/23)
- Added an option to save outputs to drive.
- Moved support button to separated and hidden section, because it looks ugly.
- Fixed some bugs where VAE can't be changed in webui.
- Updated dependencies, webui, and extensions to the latest version.
- Deleted `hitokomoru_v1_5`.
- Added [WD 1.5 Beta 2 - Aesthetic](https://huggingface.co/waifu-diffusion/wd-1-5-beta2). Releases note: [here](https://cafeai.notion.site/WD-1-5-Beta-2-Release-Notes-2852db5a9cdd456ba52fc5730b91acfd)
- Added `illuminati_diffusion_v1_1`. Required embeddings coming soon.
- Added `ref_slave_v2` and set it to default
- Used `git reset` to move the head to `3cd625854f9dc71235a432703ba82abfc5d1a3fc` when `try_new_ui_ux` set to True, as it's a stable commit history for now.
- Added `t2iadapter_sketch-fp16.safetensors` to t2iadapter model list.

### v.1.2.2 (23/02/23)
- Added **Extra** section for optional cell, the first cell added to the section is **Download Generated Images V2**, to store your output to huggingface and download it.
- Changed how the installation works. If the folder exists, then skip unpacking.
- Added `Replicant V1.0` as default model.
- Added new UI/UX theme from [Anapnoe](https://github.com/anapnoe/stable-diffusion-webui). `[Experimental]`
- Added support for `multi-controlnet` by including a slider in certain cells (default value: `2`).
- Backed up all ControlNet annotator and model data to a personal repository.

### v.1.2.1 (21/02/23)
- Added `tqdm` to track installation, unpacking, and download progress since original logs are disabled.
- Pulled the latest version of the repository and built-in extensions.
- Pre-downloaded ControlNet annotator/preprocessor.
- Added T2I Adapter model from TencentArc.
- Added several good custom upscalers such as `4x-Animesharp`, `4x-UltraSharp`, `Lollypop`, and others.
- Added Video Loopback extension for creating videos in Img2img, now with ControlNet support.
- Deleted CivitAI browser extension because another extension called `sd-filer` serves the same purpose. You can download models, lora, and embeddings from the web UI.
- Added Katanuki extension to convert results to transparent images.
- Deleted Haku Img Extensions because they were rarely used.
- Added 'Use Old Karras Scheduler' option to quick settings (header).
- Fixed an invalid URL for Chillout Mix Pruned and SD 1.5 Pruned.
- Fixed the 'Download Generated Images' cell when creating a duplicate zip from 'output.zip(1)' to 'output(n+1).zip'.
- Added checkboxes for randomly selecting VAE and Model.
