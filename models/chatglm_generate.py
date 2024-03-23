from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.configuration_chatglm import ChatGLMConfig
from torch import nn
from utils.logger import run_logger
import os
from typing import Dict
import torch
import uuid


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


class DST(nn.Module):
    def __init__(self, args, tokenizer):
        super(DST, self).__init__()

        self.args = args
        self.tokenizer = tokenizer

        config = ChatGLMConfig.from_pretrained(args.chatglm_path)
        config.n_clusters = args.n_clusters
        if args.mppt:
            config.pre_seq_len = args.pre_seq_len
            config.prefix_projection = False
        self.model = ChatGLMForConditionalGeneration.from_pretrained(args.chatglm_path, config=config)

        self.uuid = uuid.uuid1().__str__()

        if args.checkpoint and args.checkpoint != "ICL":
            run_logger.info(f"loading prefix encoder parameters from {args.checkpoint}")
            self.model.transformer.prefix_encoder.load_state_dict(
                torch.load(os.path.join(args.checkpoint, "prefix.pt")))
            with open(os.path.join(args.checkpoint, "uuid.txt"), "r") as fin:
                self.uuid = fin.read()

        if self.args.chatglm_quantize is not None:
            self.model = self.model.quantize(args.chatglm_quantize)

        self.model = self.model.half()

        if args.mppt:
            self.model.transformer.prefix_encoder.float()

    def save_model(self, path):
        save_dir = os.path.join(self.args.save_dir, path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.model.transformer.prefix_encoder.state_dict(), os.path.join(save_dir, "prefix.pt"))
        self.tokenizer.save_pretrained(save_dir)

        with open(os.path.join(save_dir, "uuid.txt"), "w") as fout:
            fout.write(self.uuid)

        run_logger.info(f"save model to: {save_dir}")

    def infer(self, data):
        inputs = data["inputs"].to("cuda")
        input_length = inputs.input_ids.shape[1]
        generate_config = {
            "num_beams": 1,
            "do_sample": False,
            "max_length": input_length + 32,
        }
        outputs = self.model.generate(**inputs, **generate_config)
        predict = outputs[:, input_length:]
        predict_state = self.tokenizer.batch_decode(predict)
        return predict_state, data

    def forward(self, data):
        output = self.model(**data["inputs"])
        return output.loss
