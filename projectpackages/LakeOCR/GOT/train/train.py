# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import pathlib
import torch
import transformers

# from GOT.train.trainer import GOTTrainer
# from GOT.train.trainer_vit_llrd import GOTTrainer
from GOT.train.trainer_vit_fixlr import GOTTrainer
from GOT.model import GOTLlamaForCausalLM
from GOT.model import *
from GOT.data import make_supervised_data_module
from GOT.utils.arguments import *
from GOT.utils.constants import *
from GOT.utils.utils import smart_tokenizer_and_embedding_resize
from GOT.model.vision_encoder.sam import build_sam_vit_b
from GOT.model.vision_encoder.swin_transformer import build_swin_transformer
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = GOTLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        '/data/hypertext/xpkong/newcode/checkpoints/kly-vary-1025-cc595-pretrain/',
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, padding_side="right", model_max_length=training_args.model_max_length,)

    # # model = AutoModelForCausalLM.from_pretrained("/data/public/ucaswei/cache/Qwen/qwen/", device_map="cuda", trust_remote_code=True).eval()

    # model = GOTQwenForCausalLM.from_pretrained(model_args.model_name_or_path, low_cpu_mem_usage=True, device_map='cuda')


    if data_args.conversation_version == "v0" or "models--decapoda-research--llama-7b-hf" in model_args.model_name_or_path:
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    # tokenizer.pad_token = DEFAULT_UNK_TOKEN
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token':'<|endoftext|>'})

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    vision_tower_dict = model.get_model().initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        freeze_vision_tower=model_args.freeze_vision_tower,
        use_im_start_end=model_args.use_im_start_end,
        vision_select_layer=model_args.vision_select_layer,
        dtype=dtype,
        device=training_args.device
    )

    model.initialize_vision_tokenizer(
        tokenizer=tokenizer, 
        freeze_lm_model=model_args.freeze_lm_model, 
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        device=training_args.device,
    )
    model.get_model().vision_tower = transformers.CLIPVisionModel.from_pretrained(
        '/data/public/ucaswei/pretrain/vit-large-patch14')
    model.get_model().vision_tower_high = build_sam_vit_b(checkpoint='/data/hypertext/xpkong/newcode/checkpoints/kly-sam-opt-all-1023-new/pytorch_model.bin')
    # model.get_model().mm_projector = create_perciever()

    model.to(dtype=dtype, device=training_args.device)
    # 'image_processor_high
    # data_args.image_token_len = vision_tower_dict['image_token_len']
    data_args.image_token_len = 256
    data_args.image_processor = vision_tower_dict['image_processor']
    data_args.image_processor_high = vision_tower_dict['image_processor_high']
    data_args.use_im_start_end = model_args.use_im_start_end

    # mixed relation, to be fixed
    if model_args.freeze_lm_model:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        # for p in model.get_model().vision_encoder.parameters():
        #     p.requires_grad = True
        # for p in model.get_model().chatt.parameters():
        #     p.requires_grad = True
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True
        # conv_final
        # for p in model.get_model().conv_final.parameters():
        #     p.requires_grad = True


        if not model_args.freeze_vision_tower:

            model.get_model().vision_tower.requires_grad_(True)
            # for i in range(20):
            #     model.get_model().vision_tower.vision_model.encoder.layers[i].requires_grad_(False)
            # model.get_model().vision_tower.vision_model.encoder.layers[-1].requires_grad_(False)
            # model.get_model().vision_tower.vision_model.embeddings.requires_grad_(False)
            # model.get_model().vision_tower.vision_model.pre_layrnorm.requires_grad_(False)
            # model.get_model().vision_tower.vision_model.post_layernorm.requires_grad_(False)

            #         for p in model.get_model().vision_encoder.parameters():
            # p.requires_grad = True

            # for n, p in model.named_parameters():
            #     print(n, p.requires_grad)

    if model_args.freeze_vision_tower:
        model.get_model().vision_tower.requires_grad_(False)
    
    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

    # params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    # if len(params_no_grad) > 0:
    #     if training_args.fsdp is not None and len(training_args.fsdp) > 0:
    #         if len(params_no_grad) < 10:
    #             print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
    #         else:
    #             print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
    #         print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
    #         print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

    #         from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    #         def patch_FSDP_use_orig_params(func):
    #             def wrap_func(*args, **kwargs):
    #                 use_orig_params = kwargs.pop('use_orig_params', True)
    #                 return func(*args, **kwargs, use_orig_params=use_orig_params)
    #             return wrap_func

    #         FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)


    data_module = make_supervised_data_module(
        interleave=training_args.interleave, 
        with_box=training_args.with_box, 
        tokenizer=tokenizer, 
        data_args=data_args
    )

    trainer = GOTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()


