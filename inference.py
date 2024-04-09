from pathlib import Path
from typing import Any, Optional, Union

import fire
import gradio as gr
import safetensors.torch
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from torchvision.utils import save_image

from model import ELLA, T5TextEmbedder


class ELLAProxyUNet(torch.nn.Module):
    def __init__(self, ella, unet):
        super().__init__()
        # In order to still use the diffusers pipeline, including various workaround

        self.ella = ella
        self.unet = unet
        self.config = unet.config
        self.dtype = unet.dtype
        self.device = unet.device

        self.flexible_max_length_workaround = None

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if self.flexible_max_length_workaround is not None:
            time_aware_encoder_hidden_state_list = []
            for i, max_length in enumerate(self.flexible_max_length_workaround):
                time_aware_encoder_hidden_state_list.append(
                    self.ella(encoder_hidden_states[i : i + 1, :max_length], timestep)
                )
            # No matter how many tokens are text features, the ella output must be 64 tokens.
            time_aware_encoder_hidden_states = torch.cat(
                time_aware_encoder_hidden_state_list, dim=0
            )
        else:
            time_aware_encoder_hidden_states = self.ella(
                encoder_hidden_states, timestep
            )

        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=time_aware_encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )


def generate_image_with_flexible_max_length(
    pipe, t5_encoder, prompt, fixed_negative=False, output_type="pt", **pipe_kwargs
):
    device = pipe.device
    dtype = pipe.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    prompt_embeds = t5_encoder(prompt, max_length=None).to(device, dtype)
    negative_prompt_embeds = t5_encoder(
        [""] * batch_size, max_length=128 if fixed_negative else None
    ).to(device, dtype)

    # diffusers pipeline concatenate `prompt_embeds` too early...
    # https://github.com/huggingface/diffusers/blob/b6d7e31d10df675d86c6fe7838044712c6dca4e9/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L913
    pipe.unet.flexible_max_length_workaround = [
        negative_prompt_embeds.size(1)
    ] * batch_size + [prompt_embeds.size(1)] * batch_size

    max_length = max([prompt_embeds.size(1), negative_prompt_embeds.size(1)])
    b, _, d = prompt_embeds.shape
    prompt_embeds = torch.cat(
        [
            prompt_embeds,
            torch.zeros(
                (b, max_length - prompt_embeds.size(1), d), device=device, dtype=dtype
            ),
        ],
        dim=1,
    )
    negative_prompt_embeds = torch.cat(
        [
            negative_prompt_embeds,
            torch.zeros(
                (b, max_length - negative_prompt_embeds.size(1), d),
                device=device,
                dtype=dtype,
            ),
        ],
        dim=1,
    )

    images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        **pipe_kwargs,
        output_type=output_type,
    ).images
    pipe.unet.flexible_max_length_workaround = None
    return images


def load_ella(filename, device, dtype):
    ella = ELLA()
    safetensors.torch.load_model(ella, filename, strict=True)
    ella.to(device, dtype=dtype)
    return ella


def load_ella_for_pipe(pipe, ella):
    pipe.unet = ELLAProxyUNet(ella, pipe.unet)


def offload_ella_for_pipe(pipe):
    pipe.unet = pipe.unet.unet


def generate_image_with_fixed_max_length(
    pipe, t5_encoder, prompt, output_type="pt", **pipe_kwargs
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    prompt_embeds = t5_encoder(prompt, max_length=128).to(pipe.device, pipe.dtype)
    negative_prompt_embeds = t5_encoder([""] * len(prompt), max_length=128).to(
        pipe.device, pipe.dtype
    )

    return pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        **pipe_kwargs,
        output_type=output_type,
    ).images


def build_demo(ella_path, sd_path="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    ella = load_ella(ella_path, pipe.device, pipe.dtype)
    t5_encoder = T5TextEmbedder().to(pipe.device, dtype=torch.float16)

    def generate_images(
        prompt, guidance_scale, seed, num_inference_steps, size=512, _batch_size=2
    ):
        print("#" * 50)
        print(prompt)
        load_ella_for_pipe(pipe, ella)
        image_flexible = generate_image_with_flexible_max_length(
            pipe,
            t5_encoder,
            [prompt] * _batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=size,
            width=size,
            generator=[
                torch.Generator(device="cuda").manual_seed(seed + i)
                for i in range(_batch_size)
            ],
            output_type="pil",
        )
        offload_ella_for_pipe(pipe)

        image_ori = pipe(
            [prompt] * _batch_size,
            output_type="pil",
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=size,
            width=size,
            generator=[
                torch.Generator(device="cuda").manual_seed(seed + i)
                for i in range(_batch_size)
            ],
        ).images

        return image_ori, image_flexible

    with gr.Blocks() as app:
        gr.Markdown(
            """
    # ELLA-SD1.5 vs SD1.5

    [ELLA Project](https://ella-diffusion.github.io/)

    ## Notes
    
    ** short prompt also works, but the result is much better after the caption is refined. **

    ### Caption Refining with In Context Learning(ICL)

    caption refining instruction example:
    ```
    Please generate the long prompt version of the short one according to the given examples. Long prompt version should consist of 3 to 5 sentences. Long prompt version must sepcify the color, shape, texture or spatial relation of the included objects. DO NOT generate sentences that describe any atmosphere!!!

    Short: A calico cat with eyes closed is perched upon a Mercedes.
    Long: a multicolored cat perched atop a shiny black car. the car is parked in front of a building with wooden walls and a green fence. the reflection of the car and the surrounding environment can be seen on the car's glossy surface.

    Short: A boys sitting on a chair holding a video game remote.
    Long: a young boy sitting on a chair, wearing a blue shirt and a baseball cap with the letter 'm'. he has a red medal around his neck and is holding a white game controller. behind him, there are two other individuals, one of whom is wearing a backpack. to the right of the boy, there's a blue trash bin with a sign that reads 'automatic party'.

    Short: A man is on the bank of the water fishing.
    Long: a serene waterscape where a person, dressed in a blue jacket and a red beanie, stands in shallow waters, fishing with a long rod. the calm waters are dotted with several sailboats anchored at a distance, and a mountain range can be seen in the background under a cloudy sky.

    Short: A kitchen with a cluttered counter and wooden cabinets.
    Long: a well-lit kitchen with wooden cabinets, a black and white checkered floor, and a refrigerator adorned with a floral decal on its side. the kitchen countertop holds various items, including a coffee maker, jars, and fruits.

    Short: a racoon holding a shiny red apple over its head
    ```

    using: https://huggingface.co/spaces/Qwen/Qwen-72B-Chat-Demo
    got: a mischievous raccoon standing on its hind legs, holding a bright red apple aloft in its furry paws. the apple shines brightly against the backdrop of a dense forest, with leaves rustling in the gentle breeze. a few scattered rocks can be seen on the ground beneath the raccoon's feet, while a gnarled tree trunk stands nearby.
            """
        )
        with gr.Row():
            input_caption = gr.Textbox(
                value="A vivid red book with a smooth, matte cover lies next to a glossy yellow vase. The vase, with a slightly curved silhouette, stands on a dark wood table with a noticeable grain pattern. The book appears slightly worn at the edges, suggesting frequent use, while the vase holds a fresh array of multicolored wildflowers."
            )
            with gr.Column():
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=16.0, value=10, label="guidance_scale"
                )
                seed = gr.Slider(
                    minimum=1000, maximum=2**20, value=1000, label="random seed"
                )
                num_inference_steps = gr.Slider(
                    minimum=15, maximum=100, value=25, label="num_inference_steps"
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### ORIGINAL Stable Diffusion Model")
                sd_output_image_gallery = gr.Gallery(columns=2, label="ORIGINAL SD")
            with gr.Column():
                gr.Markdown(f"### ELLA")
                ella_output_image_gallery = gr.Gallery(columns=2, label="ELLA")
        submit_button = gr.Button()

        submit_button.click(
            fn=generate_images,
            inputs=[input_caption, guidance_scale, seed, num_inference_steps],
            outputs=[sd_output_image_gallery, ella_output_image_gallery],
        )

    app.queue(concurrency_count=1, api_open=False)
    app.launch(share=False)


def main(save_folder, ella_path):
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    ella = load_ella(ella_path, pipe.device, pipe.dtype)
    t5_encoder = T5TextEmbedder().to(pipe.device, dtype=torch.float16)

    # prompt from ViLG-300, PartiPrompts
    # short prompt also works, but the result is much better after the caption is refined.

    # caption refining instruction example:
    # ```
    # Please generate the long prompt version of the short one according to the given examples. Long prompt version should consist of 3 to 5 sentences. Long prompt version must sepcify the color, shape, texture or spatial relation of the included objects. DO NOT generate sentences that describe any atmosphere!!!
    #
    # Short: A calico cat with eyes closed is perched upon a Mercedes.
    # Long: a multicolored cat perched atop a shiny black car. the car is parked in front of a building with wooden walls and a green fence. the reflection of the car and the surrounding environment can be seen on the car's glossy surface.
    #
    # Short: A boys sitting on a chair holding a video game remote.
    # Long: a young boy sitting on a chair, wearing a blue shirt and a baseball cap with the letter 'm'. he has a red medal around his neck and is holding a white game controller. behind him, there are two other individuals, one of whom is wearing a backpack. to the right of the boy, there's a blue trash bin with a sign that reads 'automatic party'.
    #
    # Short: A man is on the bank of the water fishing.
    # Long: a serene waterscape where a person, dressed in a blue jacket and a red beanie, stands in shallow waters, fishing with a long rod. the calm waters are dotted with several sailboats anchored at a distance, and a mountain range can be seen in the background under a cloudy sky.
    #
    # Short: A kitchen with a cluttered counter and wooden cabinets.
    # Long: a well-lit kitchen with wooden cabinets, a black and white checkered floor, and a refrigerator adorned with a floral decal on its side. the kitchen countertop holds various items, including a coffee maker, jars, and fruits.
    #
    # Short: a racoon holding a shiny red apple over its head
    # ```
    #
    # using: https://huggingface.co/spaces/Qwen/Qwen-72B-Chat-Demo
    # got: a mischievous raccoon standing on its hind legs, holding a bright red apple aloft in its furry paws. the apple shines brightly against the backdrop of a dense forest, with leaves rustling in the gentle breeze. a few scattered rocks can be seen on the ground beneath the raccoon's feet, while a gnarled tree trunk stands nearby.

    prompt_name_examples1 = [
        ("crocodile_sweater", "Crocodile in a sweater"),
        (
            "crocodile_sweater-gpt4_refined_caption",
            "a large, textured green crocodile lying comfortably on a patch of grass with a cute, knitted orange sweater enveloping its scaly body. Around its neck, the sweater features a whimsical pattern of blue and yellow stripes. In the background, a smooth, grey rock partially obscures the view of a small pond with lily pads floating on the surface.",
        ),
        ("red_book-yellow_vase", "A red book and a yellow vase."),
        (
            "red_book-yellow_vase-gpt4_refined_caption",
            "A vivid red book with a smooth, matte cover lies next to a glossy yellow vase. The vase, with a slightly curved silhouette, stands on a dark wood table with a noticeable grain pattern. The book appears slightly worn at the edges, suggesting frequent use, while the vase holds a fresh array of multicolored wildflowers.",
        ),
        ("racoon_apple", "a racoon holding a shiny red apple over its head"),
        (
            "racoon_apple_Qwen-72B-Chat-refined",
            "a mischievous raccoon standing on its hind legs, holding a bright red apple aloft in its furry paws. the apple shines brightly against the backdrop of a dense forest, with leaves rustling in the gentle breeze. a few scattered rocks can be seen on the ground beneath the raccoon's feet, while a gnarled tree trunk stands nearby.",
        ),
    ]

    # hard example prompt.
    prompt_name_examples2 = [
        (
            "falcon_chinese",
            "a chinese man wearing a white shirt and a checkered headscarf, holds a large falcon near his shoulder. the falcon has dark feathers with a distinctive beak. the background consists of a clear sky and a fence, suggesting an outdoor setting, possibly a desert or arid region",
        ),
        (
            "wombat",
            "A close-up photo of a wombat wearing a red backpack and raising both arms in the air. Mount Rushmore is in the background",
        ),
        (
            "bakkot_AstralCodexTen_2",
            "An oil painting of a man in a factory looking at a cat wearing a top hat",
        ),
    ]

    for name, prompt in prompt_name_examples1 + prompt_name_examples2:
        print("#" * 80)
        print(f'{name}: "{prompt}"')
        _batch_size = 1
        size = 512
        seed = 1001
        prompt = [prompt] * _batch_size

        load_ella_for_pipe(pipe, ella)
        image_flexible = generate_image_with_flexible_max_length(
            pipe,
            t5_encoder,
            prompt,
            guidance_scale=12,
            num_inference_steps=50,
            height=size,
            width=size,
            generator=[
                torch.Generator(device="cuda").manual_seed(seed + i)
                for i in range(_batch_size)
            ],
        )
        image_fixed = generate_image_with_fixed_max_length(
            pipe,
            t5_encoder,
            prompt,
            guidance_scale=12,
            num_inference_steps=50,
            height=size,
            width=size,
            generator=[
                torch.Generator(device="cuda").manual_seed(seed + i)
                for i in range(_batch_size)
            ],
        )
        offload_ella_for_pipe(pipe)

        image_ori = pipe(
            prompt,
            output_type="pt",
            guidance_scale=12,
            num_inference_steps=50,
            height=size,
            width=size,
            generator=[
                torch.Generator(device="cuda").manual_seed(seed + i)
                for i in range(_batch_size)
            ],
        ).images

        print(f'save image at {save_folder / f"{name}.png"}')
        print(
            "original SD1.5\t|\tELLA-SD1.5(fixed token length)\t|\tELLA-SD1.5(flexible token length)"
        )
        save_image(
            torch.cat([image_ori, image_fixed, image_flexible], dim=0),
            save_folder / f"{name}.png",
            nrow=3,
        )


if __name__ == "__main__":
    fire.Fire(dict(test=main, demo=build_demo))
