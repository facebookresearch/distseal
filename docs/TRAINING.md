For each training setting, the corresponding YAML configuration file can be found in "configs/training". All the experiments were done on ImageNet-1K, revision 061417. 

# 1️⃣ Post-hoc watermarking and in-model watermarking for RAR

We first train the post-hoc watermarking models, then distill them into the pretrained RAR models to obtain in-model watermarked generators and latent decoders.

## Post-hoc watermarking


To train a post-hoc latent watermarker (before quantization):

```bash
OMP_NUM_THREADS=40 torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --config configs/training/autoregressive/posthoc_latent_before_quant.yaml \
    --dataset.train_dir ${PATH_TO_IMAGENET_TRAIN} \
    --dataset.val_dir ${PATH_TO_IMAGENET_VAL} \
    --output_dir ${PATH_TO_SAVE_POST_HOC_MODEL} \
    --local_rank 0 --debug_slurm 
```

To train a post-hoc latent watermarker (after quantization):

```bash
OMP_NUM_THREADS=40 torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --config configs/training/autoregressive/posthoc_latent_after_quant.yaml \
    --dataset.train_dir ${PATH_TO_IMAGENET_TRAIN} \
    --dataset.val_dir ${PATH_TO_IMAGENET_VAL} \
    --output_dir ${PATH_TO_SAVE_POST_HOC_MODEL} \
    --local_rank 0 --debug_slurm
```

As a baseline, you can also train a post-hoc pixel watermarker: 

```bash
OMP_NUM_THREADS=40 torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --config configs/training/autoregressive/posthoc_pixel.yaml \
    --dataset.train_dir ${PATH_TO_IMAGENET_TRAIN} \
    --dataset.val_dir ${PATH_TO_IMAGENET_VAL} \
    --output_dir ${PATH_TO_SAVE_POST_HOC_MODEL} \
    --local_rank 0 --debug_slurm 
```


## Distilling into the autoregressive RAR transformer

After the post-hoc latent watermarking model has been trained, we can distill it into a pretrained RAR model to produce a distilled transformer that directly generates watermarked sequences of tokens. Note that for the distillation in the transformer, only the post-hoc latent watermarking model trained before the quantization can be used.

### Preprocessing

First, the training data need to be pre-tokenized and saved to .json cache files as the training code of RAR expects the input data in this format. During the pre-tokenization, we also add watermarks to the latents using the trained post-hoc watermarking model.

```bash
PYTHONPATH=deps torchrun --nproc_per_node=8 --nnodes=1 \
    scripts/pretokenize_rar.py \
    --img_size 256 \
    --batch_size 8 \
    --ten_crop \
    --data_path ${PATH_TO_IMAGENET} \
    --tokenizer cards/maskgit_base.yaml \
    --cached_path ${PATH_TO_SAVE_NON_WATERMARKED_LATENTS} \
    --output_path ${PATH_TO_SAVE_WATERMARKED_LATENTS} \
    --wm_path ${PATH_TO_POSTHOC_WATERMARK_MODEL} \
    --scaling_w 1.5 \
    --wm_message ${PATH_TO_SAVE_WATERMARK_MESSAGE_FILE_IN_TXT_FORMAT}
```

A few notes regarding the above command:
-  `PATH_TO_IMAGENET` point to the root directory of the downloaded ImageNet-1k dataset, and not to the train split.
-  Replace `PATH_TO_POSTHOC_WATERMARK_MODEL` with the checkpoint path of the post-hoc watermarking model.
- `scaling_w` refers to the watermark strength.
-  During the distillation, a fixed secret message will be generated and saved to `PATH_TO_SAVE_WATERMARK_MESSAGE_FILE_IN_TXT_FORMAT`. This message
should be shipped later together with the watermark detector to use the `distseal.loader` APIs (see example model cards in "./cards")
- The final watermarked latents are the files in the directory path specified by `PATH_TO_SAVE_WATERMARKED_LATENTS`.
- We also generate the non watermarked latents in `PATH_TO_SAVE_NON_WATERMARKED_LATENTS`.


### Distillation

We re-use the [RAR training script](https://github.com/bytedance/1d-tokenizer/blob/main/README_RAR.md) on the tokens precomputed above. See "configs/generation/rar.yaml" for the detailed parameters.

```bash

PYTHONPATH=deps WANDB_MODE=offline accelerate launch --num_machines=4 --num_processes=32 --machine_rank=0 --main_process_ip=$(hostname) --main_process_port=29551 --same_network train_rar.py \
    config=configs/generation/rar.yaml \
    experiment.project="rar" \
    experiment.name="rar_xl" \
    experiment.output_dir="distilled_rar_xl" \
    dataset.params.pretokenization=${PATH_TO_SAVE_JSONL}/pretokenized.jsonl \
    model.generator.hidden_size=1280 \
    model.generator.num_hidden_layers=32 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=5120
```

## Distilling into the latent decoder of RAR

Below we provide the command to distill the post-hoc latent watermarker (after quantization) into the latent decoder of RAR:

```bash
python distill.py \
    config=configs/training/autoregressive/inmodel_latent_decoder.yaml \
    imagenet.data_dir=${PATH_TO_IMAGENET} \
    watermarker_ckpt_path=${PATH_TO_LATENT_POSTHOC_WATERMARK_CHECKPOINT} \
    run_dir=${PATH_TO_SAVE_DISTILLED_MODEL}
```


<br>

# 2️⃣  Post-hoc watermarking and in-model watermarking for DCAE

## Post-hoc watermarking

The training of the post-hoc watermarking models is similar to RAR, only the autoencoder and the image resolution differ. Training details can be found in "configs/training/diffusion".


To train a post-hoc latent watermarking model:

```bash
OMP_NUM_THREADS=40 torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --config configs/training/diffusion/posthoc_latent.yaml \
    --dataset.train_dir ${PATH_TO_IMAGENET_TRAIN} \
    --dataset.val_dir ${PATH_TO_IMAGENET_VAL} \
    --output_dir ${PATH_TO_SAVE_POST_HOC_MODEL} \
    --local_rank 0 --debug_slurm 
```

As a baseline at resolution 512x512, you can also train a post-hoc pixel watermarker:

```bash
OMP_NUM_THREADS=40 torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --config configs/training/diffusion/posthoc_pixel.yaml \
    --dataset.train_dir ${PATH_TO_IMAGENET_TRAIN} \
    --dataset.val_dir ${PATH_TO_IMAGENET_VAL} \
    --output_dir ${PATH_TO_SAVE_POST_HOC_MODEL} \
    --local_rank 0 --debug_slurm
```

## Distilling into the diffusion transformer

First, the training data should be precomputed as latents. So, we specify the post-hoc latent watermarker checkpoint in order to create a folder with precomputed watermarked latents and the associated watermarking message.

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/dc_ae_generate_latent.py resolution=512     image_root_path=${PATH_TO_IMAGENET_TRAIN} batch_size=64     model_name=dc-ae-f64c128-in-1.0 scaling_factor=0.2889 latent_root_path=${PATH_TO_IMAGENET_LATENTS} watermarker_path=${PATH_TO_LATENT_POSTHOC_WATERMARK_CHECKPOINT} watermarker_scaling_w=0.5
```

Then we can fine-tune (or train from scratch) a DC-AE diffusion transformer on these precomputed watermarked latents:

```bash
torchrun --nnodes=1 --nproc_per_node=8 distill.py \
    config=configs/training/diffusion/inmodel_transformer.yaml \
    latent_imagenet.data_dir=${PATH_TO_IMAGENET_LATENTS} \
    watermark_ckpt_path=${PATH_TO_LATENT_POSTHOC_WATERMARK_CHECKPOINT} \
    watermark_msg_path=${PATH_TO_MSG} \
    run_dir=${PATH_TO_SAVE_DISTILLED_MODEL}
```

Note for packaging the checkpoint of the distilled transformer. The distilled transformer checkpoint consists of both a diffusion transformer and the autoencoder. For inference using `distseal.loader` API, run the script below to process the checkpoint:
```
python scripts/package_uvit.py --ckpt ${PATH_TO_SAVE_DISTILLED_MODEL}/checkpoint.pt --save_path distilled_uvit.pt
```

### Distilling into the latent decoder

To distill the post-hoc latent watermarker into DC-AE autoencoder:

```bash
python distill.py \
    config=configs/training/diffusion/inmodel_latent_decoder.yaml \
    imagenet.data_dir=${PATH_TO_IMAGENET} \
    watermarker_ckpt_path=${PATH_TO_LATENT_POSTHOC_WATERMARK_CHECKPOINT} \
    run_dir=${PATH_TO_SAVE_DISTILLED_MODEL}
```

