export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${PYTHONPATH}:./WavTokenizer"

NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

CONFIG_PATH=/mnt/data/users/ivan.rodkin/lab/speech-armt/WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml
MODEL_PATH=/mnt/data/users/ivan.rodkin/lab/speech-armt/WavTokenizer/WavTokenizer_small_600_24k_4096.ckpt

ITERS=10000
MAX_LENGTH=8192
LR=1e-4
TBS=32
BS=4

GRAD_ACC_STEPS=$((TBS/BS))
N=14

MODEL_CFG=./configs/mamba_small.json
cd ../..
accelerate launch --num_processes $NP --config_file  ./accelerate.yaml --main_process_port $((29500+$N)) finetune_music.py \
    --output_dir ../runs/music_model_output_mamba_$N \
    --model_name mamba \
    --model_cfg $MODEL_CFG \
    --tokenizer_type wavtokenizer \
    --tokenizer_name $CONFIG_PATH,$MODEL_PATH \
    --iters $ITERS \
    --max_length $MAX_LENGTH \
    --learning_rate $LR \
    --batch_size $BS \
    --warmup_steps 100 \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --early_stopping_steps 10
