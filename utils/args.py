import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sgd_data_dir", default="./data/sgd", type=str)
parser.add_argument("--crosswoz_data_dir", default="./data/crosswoz", type=str)
parser.add_argument("--risawoz_data_dir", default="./data/risawoz", type=str)
parser.add_argument("--multiwoz2_0_data_dir", default="./data/multiwoz2.0", type=str)
parser.add_argument("--multiwoz2_1_data_dir", default="./data/multiwoz2.1", type=str)
parser.add_argument("--multiwoz2_2_data_dir", default="./data/multiwoz2.2", type=str)
parser.add_argument("--multiwoz2_3_data_dir", default="./data/multiwoz2.3", type=str)
parser.add_argument("--multiwoz2_4_data_dir", default="./data/multiwoz2.4", type=str)
parser.add_argument("--all_data_dir", default="./data/all", type=str)

parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--n_epochs", default=12, type=int)
parser.add_argument("--eval_epoch", default=1, type=int)
parser.add_argument("--save_epoch", default=1, type=int)
parser.add_argument("--n_clusters", default=1, type=int)

parser.add_argument("--word_dropout", default=0.1, type=float)

parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--patience", default=8, type=int)
parser.add_argument("--attn_head", default=12, type=int)
parser.add_argument("--num_history", default=20, type=int)
parser.add_argument(
    "--distance_metric", default="euclidean", type=str, help="euclidean or cosine"
)

parser.add_argument("--num_self_attention_layer", default=6, type=int)

parser.add_argument("--dataset", default="sgd", type=str, help="woz2.1~2.4 or sgd")

parser.add_argument(
    "--nhead", default=12, type=int, help="number of generator decoder layer"
)
parser.add_argument("--schema_dir", default="./data", type=str, help="schema dir")
# parser.add_argument(
#     "--model_name_or_path",
#     default="/home/twtang/model/bert-base",
#     type=str,
#     help="pretrained model name or path",
# )

parser.add_argument(
    "--model_name_or_path",
    default="/public/home/htwang/twtang/models/t5-small",
    type=str,
    help="pretrained model name or path",
)

parser.add_argument("--train_batch_size", default=1, type=int, help="train batch size")
parser.add_argument("--evaluate_batch_size", default=1, type=int, help="train batch size")

parser.add_argument("--epoches", default=10, type=int, help="train epoches")
parser.add_argument(
    "--dropout_prob", default=0.1, type=float, help="dropout probability"
)
parser.add_argument(
    "--num_decoder_layers", default=6, type=int, help="decode layer numbers"
)

parser.add_argument("--max_predict_length", default=32, type=int, help="max predict length")
parser.add_argument("--ngpu", default=1, type=int, help="number of gpus")
parser.add_argument("--max_predict_len", default=32, type=int, help="max predict len")
parser.add_argument("--save_dir", default=f"save", type=str,
                    help="model save path")

parser.add_argument("--multi_gpu", action="store_true", help="use multi gpu")
parser.add_argument("--parallel_decode", action="store_true", help="whether use parallel decoder")
parser.add_argument("--use_cuda", action="store_true", help="use cuda or not")
parser.add_argument("--train_exist", action="store_true", help="only generate exist slot value instead of none")
parser.add_argument("--show_prediction", action="store_true", help="show predicted dialogue state")
parser.add_argument("--save_model", action="store_true", help="save model")
parser.add_argument("--use_previous_state", action="store_true", help="use previous dialogue state")
parser.add_argument("--plot_loss", action="store_true", help="plot train loss")
parser.add_argument("--debug", action="store_true", help="debug model")
parser.add_argument("--use_single_state", action="store_true", help="train one utterance with only single state")
parser.add_argument("--less_tqdm_output", action="store_true", help="show less tqdm output")
parser.add_argument("--save_predict_errors", action="store_true", help="save all prediction errors")
parser.add_argument("--save_to_csv", action="store_true", help="save result to csv")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--ignore_pad_token_for_loss", action="store_true")
parser.add_argument("--mppt", action="store_true", help="multiple prefix prompt tuning")
parser.add_argument("--save_prediction", action="store_true")
parser.add_argument("--use_lower", action="store_true")
parser.add_argument("--zero_shot", action="store_true")
parser.add_argument("--clean_data", action="store_true")
parser.add_argument("--ICL", action="store_true")
parser.add_argument("--use_all_state", action="store_true")
parser.add_argument("--random_prompt", action="store_true")
parser.add_argument("--ensemble_inference", action="store")


parser.add_argument("--checkpoint", default="ICL", type=str, help="model check point")
parser.add_argument("--backbone", default="glm", type=str, help="backbone of dst model")
parser.add_argument("--glm_path", default="/data/twtang/models/glm-2b", type=str, help="glm model path")
parser.add_argument("--t5_path", default="/data/twtang/models/t5-small", type=str, help="t5-small model path")
parser.add_argument("--gpt2_path", default="/data/twtang/models/gpt2", type=str, help="gpt2 model path")
parser.add_argument("--llama2_path", default="/public/home/wlchen/plm/Llama-2-7b-chat-hf", type=str,
                    help="llama-2-7b-chat model path")
parser.add_argument("--chatglm_path", default="/public/home/wlchen/plm/chatglm-6b", type=str,
                    help="chatglm-6b model path")
parser.add_argument("--chatglm2_path", default="/public/home/wlchen/plm/chatglm2-6b", type=str,
                    help="chatglm2-6b model path")
parser.add_argument("--chatglm_int4_path", default="/data/twtang/models/chatglm-6b-int4", type=str,
                    help="chatglm-6b-int4 model path")
parser.add_argument("--chatglm_int8_path", default="/data/twtang/models/chatglm-6b-int8", type=str,
                    help="chatglm-6b-int8 model path")
parser.add_argument("--chatglm_quantize", default=None, type=int, help="chatglm qunantize")

parser.add_argument("--input_context", default="slotname-utterance", type=str,
                    help="choice: utterance, utterance-single-current-state, utterance-all-current-state, utterance-previous-current-sate")
parser.add_argument("--num_sample", default=0, type=int, help="data sampler number")
parser.add_argument("--data_ratio", default=100, type=int, help="data train ratio")
parser.add_argument("--prediction_save_path", default="./prediction", type=str,
                    help="prediction error save path")
parser.add_argument("--local_rank", default=0, type=int, help="process local rank")
parser.add_argument("--mp_size", default=4, type=int, help="model parallel size")
parser.add_argument("--pre_seq_len", default=10, type=int)
parser.add_argument("--max_source_length", default=512, type=int)
parser.add_argument("--max_target_length", default=32, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--result_log", default="./logging/result.log", type=str)
parser.add_argument("--run_log", default="./logging/run.log", type=str)
parser.add_argument("--history_turn", default=1, type=int)
parser.add_argument("--result_csv_dir", default="result", type=str)
parser.add_argument(
    "--eval_step",
    default=2000,
    type=int,
    help="Within each epoch, do evaluation as well at every eval_step",
)
parser.add_argument("--train_id", default="1", type=str)
parser.add_argument("--total_step", default=50000, type=int)
parser.add_argument("--save_step", default=2000, type=int)
parser.add_argument("--exclude_domain", default="all", type=str)
parser.add_argument("--cluster_feature", default="transformer", type=str)
parser.add_argument("--none_rate", default=1, type=float)

args = parser.parse_args()
