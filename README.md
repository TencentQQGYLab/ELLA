# ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment

<div align="center">
<span class="author-block">
    <a href="https://openreview.net/profile?id=~Xiwei_Hu1">Xiwei Hu*</a>,
</span>
<span class="author-block">
    <a href="https://wrong.wang/">Rui Wang*</a>,
</span>
<span class="author-block">
    <a href="https://openreview.net/profile?id=~Yixiao_Fang1">Yixiao Fang*</a>,
</span>
<span class="author-block">
    <a href="https://openreview.net/profile?id=~BIN_FU2">Bin Fu*</a>,
</span>
<span class="author-block">
    <a href="https://openreview.net/profile?id=~Pei_Cheng1">Pei Cheng</a>,
</span>
<span class="author-block">
    <a href="https://www.skicyyu.org/">Gang Yu&#10022</a>
</span>
<p>
* Equal contributions, &#10022 Corresponding Author
</p>

<img src="./assets/ELLA-Diffusion.jpg" width="30%" > <br/>
<a href='https://ella-diffusion.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2403.05135'><img src='https://img.shields.io/badge/arXiv-2403.05135-b31b1b.svg'></a>
</div>

Official code of "ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment".
<p>
</p>
<div align="center">
    <img src="./assets/teaser_3img.png" width="100%">
    <img src="./assets/teaser1_raccoon.png" width="100%">
</div>

## 🌟 Changelog

- **[2024.4.9]** 🔥🔥🔥 Release [ELLA-SD1.5](https://huggingface.co/QQGYLab/ELLA/blob/main/ella-sd1.5-tsc-t5xl.safetensors) Checkpoint! Welcome to try! 
- **[2024.3.11]** 🔥 Release DPG-Bench! Welcome to try! 
- **[2024.3.7]** Initial update

## Inference

### ELLA-SD1.5


```bash
# get ELLA-SD1.5 at https://huggingface.co/QQGYLab/ELLA/blob/main/ella-sd1.5-tsc-t5xl.safetensors

# comparing ella-sd1.5 and sd1.5
# will generate images at `./assets/ella-inference-examples`
python3 inference.py test --save_folder ./assets/ella-inference-examples --ella_path /path/to/ella-sd1.5-tsc-t5xl.safetensors

# build a demo for ella-sd1.5
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=8082 python3 ./inference.py demo /path/to/ella-sd1.5-tsc-t5xl.safetensors
```


## 📊 DPG-Bench

The guideline of DPG-Bench:

1. Generate your images according to our [prompts](./dpg_bench/prompts/).
    
    It is recommended to generate 4 images per prompt and grid them to 2x2 format. **Please Make sure your generated image's filename is the same with the prompt's filename.**

2. Run the following command to conduct evaluation.

    ```bash
    bash dpg_bench/dist_eval.sh $YOUR_IMAGE_PATH $RESOLUTION
    ```

Thanks to the excellent work of [DSG](https://github.com/j-min/DSG) sincerely, we follow their instructions to generate questions and answers of DPG-Bench.

## 📝 TODO

- [ ] add huggingface demo link
- [x] release checkpoint
- [x] release inference code
- [x] release DPG-Bench


## 💡 Others

We have also found [LaVi-Bridge](https://arxiv.org/abs/2403.07860), another independent but similar work completed almost concurrently, which offers additional insights not covered by ELLA. The difference between ELLA and LaVi-Bridge can be found in [issue 13](https://github.com/ELLA-Diffusion/ELLA/issues/13). We are delighted to welcome other researchers and community users to promote the development of this field.

## 😉 Citation

If you find **ELLA** useful for your research and applications, please cite us using this BibTeX:

```
@misc{hu2024ella,
      title={ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment}, 
      author={Xiwei Hu and Rui Wang and Yixiao Fang and Bin Fu and Pei Cheng and Gang Yu},
      year={2024},
      eprint={2403.05135},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
