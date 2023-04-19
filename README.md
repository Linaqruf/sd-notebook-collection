# **Cagliostro Colab UI**
All-in-One, Customizable and Flexible AUTOMATIC1111's Stable Diffusion Web UI for Google Colab. <br>
 
## What's New?
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
