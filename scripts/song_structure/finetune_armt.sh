export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:./WavTokenizer"
export HF_Trainer=1

NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

CONFIG_PATH=/mnt/data/users/ivan.rodkin/lab/speech-armt/WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml
MODEL_PATH=/mnt/data/users/ivan.rodkin/lab/speech-armt/WavTokenizer/WavTokenizer_small_600_24k_4096.ckpt

ITERS=10000
MAX_LENGTH=8192
LR=1e-4
TBS=32

BS=4

MEMORY_SIZE=32
D_MEM=64
SEGMENT_SIZE=1024

N=2

GRAD_ACC_STEPS=$((TBS/BS))

MODEL_CFG=./configs/gptneox_small.json
cd ../..
accelerate launch --num_processes $NP --config_file  ./accelerate.yaml --main_process_port $((29500+$N)) finetune_music.py \
    --model_cfg $MODEL_CFG \
    --model_name armt \
    --tokenizer_type wavtokenizer \
    --tokenizer_name $CONFIG_PATH,$MODEL_PATH \
    --iters $ITERS \
    --max_length $MAX_LENGTH \
    --learning_rate $LR \
    --batch_size $BS \
    --warmup_steps 100 \
    --num_mem_tokens $MEMORY_SIZE \
    --d_mem $D_MEM \
    --segment_size $SEGMENT_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS

