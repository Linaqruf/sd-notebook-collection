# Cagliostro Colab UI User Manual

Welcome to the [Cagliostro Colab UI](https://colab.research.google.com/github/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb) User Manual! This guide will walk you through the basics of using Cagliostro Colab UI, an innovative and powerful notebook designed to launch [Automatic1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) in Google Colab. With its advanced features, customizability, and flexibility, Cagliostro Colab offers a seamless and efficient way to utilize Stable Diffusion Web UI for your projects.

To get started with Cagliostro Colab UI, you'll need to have a Google account. Once you're logged in, you can open up Google Colab by visiting the website and signing in with your Google credentials.

## Table of Contents
- [Cagliostro Colab UI User Manual](#cagliostro-colab-ui-user-manual)
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Main Cell Explained](#main-cell-explained)
    - [Install Stable Diffusion Web UI](#install-stable-diffusion-web-ui)
    - [Download Model and VAE](#download-model-and-vae)
    - [ControlNet V1.1](#controlnet-v11)
    - [Custom Download Corner](#custom-download-corner)
    - [Start Stable Diffusion Web UI](#start-stable-diffusion-web-ui)
    - [Download Generated Images](#download-generated-images)
- [Extra Cell Explained](#extra-cell-explained)
    - [Download Generated Images V2](#download-generated-images-v2)
- [Resources](#resources)
    - [Default Extensions](#default-extensions)
    - [Default Negative Embeddings](#default-negative-embeddings)
    - [Default Custom Upscalers](#default-custom-upscalers)
    - [Theme Selector](#theme-selector)

## Features:
| Feature                                 | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Faster installation**                  | Unpacks a pre-installed repository instead of installing from scratch, saving time and reducing the risk of installation errors.Uses pre-installed Python dependencies and unpacks them to `/usr/local/lib/python3.9/dist-packages/` instead of downloading them during installation, further reducing installation time.                                                                                                                                                              |
| **Faster downloader**                       | Uses the `aria2c` downloader instead of `wget`.Downloads packages with up to 16 parallel connections, 16 threads per connection, and a 1MB chunk size for faster downloads.Used to download pre-installed repo, dependencies, and models.                                                                                                                                                                                                                       |
| **Better UI/UX**                            | Uses [Anapnoe's](https://github.com/anapnoe/stable-diffusion-webui-ux) forked version of [Automatic1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) that focuses on developing better UI/UX for Stable Diffusion Web UI. But users can still use [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) by disabling the `use_anapnoe_ui` option.Note that Anapnoe's still uses an old commit, so if users want to experience Gradio 3.23.0, it's better to disable the `use_anapnoe_ui` option. |
| **Up-to-date**                              | Enabled with `git pull` for both the repository and all extensions by default to increase user experience.Some extensions might be skipped from updating due to upgrading its Gradio to 3.23.                                                                                                                                                                                                                                                                                   |
| **Integrated to Google Drive**              | Provides a way to enhance the user experience for `Google Drive` users.Includes options to mount Google Drive, save output to Google Drive, load the model directory from Google Drive by using `unionfs-fuse` to merge folder A and folder B to folder C, and load the model file from Google Drive by copying the model to [Automatic1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/).                               |
| **Colab Optimization**                      | Loads the model in VRAM, merges in VRAM, and uses VRAM instead of CPU for most things to create a better environment for Colab free tier users.                                                                                                                                                                                                                                                                                                                               |
| **Better Default Model**                    | Provides the top trending default model and the best choice in quality.Chooses Huggingface server instead of CivitAI because it's more stable and faster.                                                                                                                                                                                                                                                                                                               |
| **All-in-One ControlNet model**             | Provides not only the ControlNet model from SD1.5 but also SD2.1.Users can set the max model, config, and adapter config before starting the Web UI.                                                                                                                                                                                                                                                                                                                     |
| **Advanced Custom Download**                 | Provides seven fields to download models, including custom models, VAE, embedding, LoRA, hypernetwork, control, and extension.Users can prune their model by typing `fp16:url` to prune with FP16 precision and `fp32:url` for FP32 precision.Users can also easily load models from Google Drive by merging folders with the `fuse:path` command.                                                                                                                            |
| **More Tunnels!**                           | Uses `sd-webui-tunnels` extension (forked by [Camenduru](https://github.com/camenduru/sd-webui-tunnels)) and set `--multiple` by default. No more boring `gradio.live`!We're also support `ngrok`.

## Main Cell Explained
### Install Stable Diffusion Web UI
This cell installs stable-diffusion-webui repository in Colab. It also includes several configuration options, such as mounting Google Drive, updating extensions, and choosing between different versions of the web UI. 

Options                                | Default Value         | Description
---------------------------------------|-----------------------|---------------------------------------------------------------
Drive Config                           |                       | 
`mount_drive`                          | `False`               | Mount your Google Drive to `/content/drive/MyDrive/` to load models.
`output_to_drive`                      | `False`               | Save generation outputs to your Google Drive instead of `/content/stable-diffusion-webui/outputs`.
`output_drive_folder`                  | `False`               | Set your default drive folder name to save your generation outputs, default: `cagliostro-colab-ui/outputs`
Web UI Config                          |                       | 
`use_anapnoe_ui`                       | `True`                | Use Anapnoe's forked repository instead of Automatic1111's.
`update_webui`                         | `True`                | Update the web UI to the latest version. Because we're using pre-installed repo, this option is crucial if you want to try latest update.
`update_extensions`                    | `True`                | Update all extensions to the latest version.
`commit_hash`                          | `''`                  | Go back to a specific commit hash to prevent errors when the web UI is updated.
`colab_optimizations`                  | `False`               | Load SDv2 the model into VRAM, optimizes demo.queue() for Colab, and loads the model onto GPU memory if available.


Note: 
- The options `colab_optimizations` are not relevant anymore as the new argument `--lowram` serves the same purpose, allowing the model to be loaded onto the GPU memory instead of VRAM if available. It also not recommended to set this `True` if you have Colab Pro subscription.

### Download Model and VAE
This code block is responsible for downloading pre-trained models and VAEs (Variational Autoencoders) from Hugging Face's model hub. The available models and VAEs are listed as boolean checkboxes, and the user can select the ones they want to download.  

| Model Name | Description | Model Page |
| --- | --- | --- |
| Stable Diffusion V1.x Model |  |  |
| `anything_v3_0` | State-of-the-art ~~merged/overfitted/vae broken~~ model when it was first launched in November, it has influenced most anime models since then. | [Link](https://huggingface.co/AdamOswald1/Anything-Preservation/blob/4121e81acc47bb87e46480ba1344b5ab57134b88/Anything-V3.0-pruned.safetensors) |
| `anime_pastel_dream` | The best alternative model to `pastelmix`, `anime_pastel_dream` was developed to provide a better user experience. | [Link](https://huggingface.co/Lykon/AnimePastelDream/blob/main/AnimePastelDream_Soft_noVae_fp16.safetensors) |
| `anylora` (default) | The best alternative model to `anything_v4_5`, `anylora` was developed to provide a better user experience. | [Link](https://huggingface.co/Lykon/AnyLoRA/blob/main/AnyLoRA_noVae_fp16.safetensors) |
| `chilloutmix_ni` | The most used non-anime model for now, it generates real person pictures but is still influenced by anime models. | [Link](https://huggingface.co/naonovn/chilloutmix_NiPrunedFp32Fix/blob/main/chilloutmix_NiPrunedFp32Fix.safetensors) |
| Stable Diffusion V2.x Model |  |  |
| `replicant_v2` | An anime model based off Waifu Diffusion V1.5 Beta 2, which performs better than `waifu_diffusion_v1_5_e2_aesthetic`. | [Link](https://huggingface.co/gsdf/Replicant-V2.0/blob/main/Replicant-V2.0_fp16.safetensors) |
| `waifu_diffusion_v1_5_e2_aesthetic` | An overtrained version of Waifu Diffusion V1.5 Beta 2 trained off stable diffusion v2.1 768. It works great to generate anime art. | [Link](https://huggingface.co/waifu-diffusion/wd-1-5-beta2/blob/main/checkpoints/wd-1-5-beta2-aesthetic-fp16.safetensors) |
| VAE models | | |
| `anime` (default) | A popular anime VAE used by many model, such as AbyssOrangeMixs and Anything V3. | 🔐 |
| `waifu_diffusion` | Anime VAE developed by the Waifu Diffusion developer based on the stable diffusion VAE. It provides contrast color compared to the `anime` VAE. | [Link](https://huggingface.co/hakurei/waifu-diffusion-v1-4/blob/main/vae/kl-f8-anime.ckpt) |
| `stable_diffusion` | The original Stable Diffusion VAE trained by Stability AI. It is intended to be used with the original CompVis Stable Diffusion. | [Link](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.ckpt) |

### ControlNet V1.1
[ControlNet](https://github.com/lllyasviel/ControlNet) is a neural network structure to control diffusion models by adding extra conditions. ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang and Maneesh Agrawala. It introduces a framework that allows for supporting various spatial contexts that can serve as additional conditionings to Diffusion models such as Stable Diffusion. It basically allow user to control Stable Diffusion generation. ControlNet 1.1 has been released nightly at this [github repository](https://github.com/lllyasviel/ControlNet-v1-1-nightly) and Cagliostro Colab UI is now support it!

| Option | Default | Description |
|--------|---------|-------------|
| `pre_download_annotator` | `True` | Specifies whether to download ControlNet pre-processor/annotator before starting the Web UI. The pre-processor/annotator includes 13 items in total, and will be downloaded to the specified path. |
| `control_v11_sd15_model` | `True` | Specifies whether to download all 14 extracted [ControlNet v1.1](https://github.com/lllyasviel/ControlNet) model developed by [Lvmin Zhang](https://twitter.com/lvminzhang) |
| `t2i_adapter_sd15_model` | `False` | Specifies whether to download all 12 [Text to Image Adapter](https://github.com/TencentARC/T2I-Adapter) model developed by [TencentArcLab](https://github.com/TencentARC). |
| `control_v10_sd21_model` | `False` | Specifies whether to download all 5 extracted [SDv21 ControlNet v1.0](https://huggingface.co/thibaud/controlnet-sd21) model developed by [thibaud](https://twitter.com/thibaudz). |
| `control_v10_wd15_model` | `False` | Specifies whether to download all 3 extracted [Waifu Diffusion 1.5 Controlnet v1.0](https://huggingface.co/furusu/ControlNet) model developed by [furusu](https://twitter.com/gcem156). |
| `control_net_max_models_num` | `2` | Specifies the maximum number of ControlNet tabs for multi-controlnet generation. |
| `control_net_model_adapter_config` | `sketch_adapter_v14.yaml` | Specifies the configuration file to use for the ControlNet model adapter. Different models require different configurations. The default configuration is `sketch_adapter_v14.yaml`. |

[T2I Adapter](https://github.com/TencentARC/T2I-Adapter) is a simple and small **(~70M parameters, ~300M storage space)** network that can provide extra guidance to pre-trained text-to-image models while freezing the original large text-to-image models. Controlnet was introduced in [T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://huggingface.co/TencentARC/T2I-Adapter). It basically a cheapest ControlNet that allow user to control Text to Image generation.

| `t2i_adapter_model` | `control_net_model_adapter_config` |
|:-------------------------:|:-------------------------:|
| t2iadapter_canny_sd14v1.pth | sketch_adapter_v14.yaml |
| t2iadapter_sketch_sd14v1.pth | sketch_adapter_v14.yaml |
| t2iadapter_seg_sd14v1.pth | image_adapter_v14.yaml |
| t2iadapter_keypose_sd14v1.pth | image_adapter_v14.yaml |
| t2iadapter_openpose_sd14v1.pth | image_adapter_v14.yaml |
| t2iadapter_color_sd14v1.pth | t2iadapter_color_sd14v1.yaml |
| t2iadapter_style_sd14v1.pth | t2iadapter_style_sd14v1.yaml |

### Custom Download Corner
This is a cell that allows you to download custom `models`, `VAEs`, `embeddings`, `LoRA`, `hypernetworks`, `upscalee` and install `extensions` by providing URLs to the files you want to download. This cell downloads custom files from various sources, including Google Drive, Huggingface, CivitAI, and other direct download links.

| Feature | Description | How to Use | Example |
| --- | --- | --- | --- |
| **Multiple Downloads** | Download multiple files at once. | Fill in the URL fields with the links to the files you want to download. Separate multiple URLs with a comma. | `url1, url2, url3` |
| **Auto-prune** | Prune models after downloading | Add `fp16:` or `fp32:` before URLs. | `fp16:url1` |
| **Copy from Google Drive** | Copy models from Google Drive and load them in the session. | Make sure you have already mounted Google Drive. Type the path to your model/lora/embedding from Google Drive. | `/content/drive/MyDrive/path/to/folder` |
| **Fusing Folder** | Fuse models/embeddings/LoRA folder to `/content/fused/{category}`. | Make sure you have already mounted Google Drive. Add `fuse:` before the path to the folder. | `fuse:/path/to/gdrive/folder` |
| **Auto-extract** | Extract files after downloading. | Add `links/to/file` ending with `.zip` or `.tar.lz4`. Extract files to specified destination directory. | `https//link.com/to/file` |
| **Install Extensions** | Install extensions for Stable Diffusion Web UI. | Add the link to the Github repository to `custom_extension_url`. | `https://github.com/user/repo` |

Once the download is complete, you can proceed to the next step.

### Start Stable Diffusion Web UI
This is a cell for launching Stable Diffusion Web UI after a long configuration process above. However, it still has its own configuration requirements, such as defining arguments, using alternative tunnels, selecting a theme, and so on.

Option          | Default  | Description
----------------|----------|-------------
| Alternative Tunnels |  | Note: Recommended Tunnels: `ngrok` > `cloudflared` > `remotemoe` > `localhostrun` > `googleusercontent` > `gradio` |
`tunnel`           | `multiple`  | Allow users to use alternative tunnels for shared links such as `cloudflared`, `remotemoe`, `localhostrun`, and `googleusercontent`.
`ngrok_token`      | `' '` | Enable ngrok tunnel for shared links. Users can get their ngrok token from [here](https://dashboard.ngrok.com/get-started/your-authtoken). If `ngrok_token` is enabled, it will automatically disable `alt_tunnels`. 
`ngrok_region`   | `ap`     | Specify the desired region for ngrok tunnel. Users can choose between `["us", "eu", "au", "ap", "sa", "jp", "in"]`.
| Launch Arguments |  |  |
| `theme_selector` | `ogxBGreen` | This section used to change preferred theme for [Anapnoe's Stable Diffusion Web UI/UX](https://github.com/anapnoe/stable-diffusion-webui-ux) |
| `use_gradio_auth` | `False` | If enabled, every time a user opens shared links, it will ask for authentication. The username is set to `cagliostro` and the password will be randomly generated with 6 ASCII characters and numbers. Authentication information will be printed when the user launches the Web UI. |
| `accelerator` | `xformers` | To accelerate your generation and training, we offer 3 arguments. People commonly use `xformers`, but `opt-sdp-attention` would be a nice choice to have after Colab updates PyTorch to 2.0. It's claimed to be better than `xformers` but only for generating at lower resolution. So we choose to use `xformers` by default. |
| `quiet_mode` | `True` | This is a quality-of-life option. If enabled, it will use `--no-hashing` to skip model hashing, which is useful for not double-checking the model if it doesn't have the same hash and for merging the model. If enabled, it will also use `--disable-console-progressbars` to only have 1 line progress bars instead of updating new lines every generation. |
| `auto_select_model` | `False` | This argument allows searching for a model and randomly selecting the model in the model directory. |
| `auto_select_VAE` | `True` | This argument allows searching for a VAE and randomly selecting the VAE in the VAE directory. |
| `no_half_VAE` | `True` | This argument prevents black generation caused by certain VAEs. This usually happens because certain VAEs like `anime` have FP32. |
| `additional_args` | `--no-download-sd-model --gradio-queue` | This argument allows the user to add custom args from [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings). |

### Download Generated Images
This cell useful to download outputs generated from Stable Diffusion Web UI as a compressed `.zip`

| Option      | Default | Description                                                                                              |
| ----------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `use_drive` | `True`    | If enabled, this option will ask to mount your Google Drive account and save the compressed file to a preferred folder name specified by `folder_name`. It will also provide a download link after the process is completed. |
| `folder_name` | `"AI-generated Art"` | This option is used to specify the preferred folder name in Google Drive if `use_drive` is enabled. |
| `filename`  | `"waifu.zip"` | This option allows the user to specify the filename of the zipfile. If `use_drive` is enabled and a duplicate filename is found in Google Drive, it will be renamed to `filename(n+1).zip`. |

## Extra Cell Explained
### Download Generated Images V2
Same as [Download Generated Images](#download-generated-images), the difference is it send the zipfile to Huggingface instead of Google Drive. Personally, it's better than save it to Google Drive.

| Option          | Default Value | Description                                                                                                       |
|-----------------|---------------|-------------------------------------------------------------------------------------------------------------------|
| `write_token`   | `""  `          | User needs to specify a `WRITE` token from [Hugging Face settings](https://huggingface.co/settings/tokens).        |
| `repo_name`     | `"ai-art-dump"` | Specifies where the repository is located. If no repository is available, one will be created automatically.      |
| `private_repo`  | `False`         | Specifies whether the repository is private or public to everyone.                                               |
| `project_name`  | `"waifu"`       | Same as [Download Generated Images](#download-generated-images), this option allows the user to specify the name of the zipfile. |

## Resources
### Default Extensions
| Extensions                                           | Github Repository                                                |
| ----------------------------------------------- | --------------------------------------------------- |
| `ashen-sensored/sd_webui_stealth_pnginfo`         | [Link](https://github.com/ashen-sensored/sd_webui_stealth_pnginfo)             |
| `hnmr293/sd-webui-cutoff`                         | [Link](https://github.com/hnmr293/sd-webui-cutoff)                             |
| `KohakuBlueleaf/a1111-sd-webui-locon`             | [Link](https://github.com/KohakuBlueleaf/a1111-sd-webui-locon)                 |
| `DominikDoom/a1111-sd-webui-tagcomplete`          | [Link](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git)         |
| `etherealxx/batchlinks-webui`                     | [Link](https://github.com/etherealxx/batchlinks-webui)                         |
| `mcmonkeyprojects/sd-dynamic-thresholding`        | [Link](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding)           |
| `kohya-ss/sd-webui-additional-networks`           | [Link](https://github.com/kohya-ss/sd-webui-additional-networks.git)         |
| `thomasasfk/sd-webui-aspect-ratio-helper`         | [Link](https://github.com/thomasasfk/sd-webui-aspect-ratio-helper.git)       |
| `Mikubill/sd-webui-controlnet`                    | [Link](https://github.com/Mikubill/sd-webui-controlnet)                       |
| `camenduru/sd-webui-tunnels`                       | [Link](https://github.com/camenduru/sd-webui-tunnels)                         |
| `bbc-mc/sdweb-merge-block-weighted-gui`           | [Link](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui.git)         |
| `bbc-mc/sdweb-xyplus`                             | [Link](https://github.com/bbc-mc/sdweb-xyplus)                                 |
| `opparco/stable-diffusion-webui-composable-lora`  | [Link](https://github.com/opparco/stable-diffusion-webui-composable-lora.git) |
| `AlUlkesh/stable-diffusion-webui-images-browser`  | [Link](https://github.com/AlUlkesh/stable-diffusion-webui-images-browser.git) |
| `arenatemp/stable-diffusion-webui-model-toolkit`  | [Link](https://github.com/arenatemp/stable-diffusion-webui-model-toolkit)     |
| `opparco/stable-diffusion-webui-two-shot`          | [Link](https://github.com/opparco/stable-diffusion-webui-two-shot)           |
| `Coyote-A/ultimate-upscale-for-automatic1111`      | [Link](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git)   |

### Default Custom Upscalers
| ESRGAN Upscaler | Mirror Link |
|------|------|
| `lollypop.pth` | [Link](https://huggingface.co/Linaqruf/stolen/blob/main/upscaler/lollypop.pth) |
| `4x-AnimeSharp.pth` | [Link](https://huggingface.co/Linaqruf/stolen/blob/main/upscaler/4x-AnimeSharp.pth) |
| `4x_foolhardy_Remacri.pth` | [Link](https://huggingface.co/Linaqruf/stolen/blob/main/upscaler/4x_foolhardy_Remacri.pth) |
| `4x-UltraSharp.pth` | [Link](https://huggingface.co/Linaqruf/stolen/blob/main/upscaler/4x-UltraSharp.pth) |

### Default Negative Embeddings 
|  Negative Embeddings                      | Link                                                                                      |
|-------------------------------------------|-------------------------------------------------------------------------------------------|
| SDv1.x  Negative Embeddings | |
| `EasyNegative.safetensors `                 | [Link](https://huggingface.co/datasets/gsdf/EasyNegative/blob/main/EasyNegative.safetensors)                  |
| `bad-artist-anime.pt`                       | [Link](https://huggingface.co/nick-x-hacker/bad-artist/blob/main/bad-artist-anime.pt)                        |
| `bad-hands-5.pt`                            | [Link](https://huggingface.co/embed/negative/blob/main/bad-hands-5.pt)                            |
| `bad-artist.pt`                             | [Link](https://huggingface.co/nick-x-hacker/bad-artist/blob/main/bad-artist.pt)                             |
| `bad_prompt_version2.pt`                     | [Link](https://huggingface.co/datasets/Nerfgun3/bad_prompt/blob/main/bad_prompt_version2.pt)                     |
| `bad_prompt.pt`                             | [Link](https://huggingface.co/datasets/Nerfgun3/bad_prompt/blob/main/bad_prompt.pt)                             |
| `bad-image-v2-39000.pt`                     | [Link](https://huggingface.co/Xynon/models/blob/main/experimentals/TI/bad-image-v2-39000.pt)                     |
| `ng_deepnegative_v1_75t.pt`                 | [Link](https://huggingface.co/embed/negative/blob/main/ng_deepnegative_v1_75t.pt)                 |
| SDv2.x  Negative Embeddings | |
| `rev2-badprompt.safetensors`                | [Link](https://huggingface.co/gsdf/Replicant-V2.0/blob/main/rev2-badprompt.safetensors)                |
| `re-badprompt.safetensors`                  | [Link](https://huggingface.co/gsdf/Replicant-V1.0/blob/main/re-badprompt.safetensors)                  |
| `wdbadprompt.pt`                            | [Link](https://huggingface.co/waifu-diffusion/wd-1-5-beta/blob/main/embeddings/wdbadprompt.pt)        |
