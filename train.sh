export BS=3; rm -r output_dir; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../src USE_TF=0 deepspeed  \
./finetune_trainer.py --model_name_or_path "" --base_model_path "EleutherAI/gpt-j-6B" --output_dir output_dir --adam_eps 1e-06 \
--data_dir "webnlg-dataset/" --do_train --evaluation_strategy=steps  --freeze_embeds --label_smoothing 0 \
--n_prompt_tokens 50 --n_prompts 2 --learning_rate 1e-01 --weight_decay 0.0 --logging_first_step --logging_steps 1000 --max_source_length 512 \
--num_train_epochs 1 --overwrite_output_dir --per_device_eval_batch_size $BS --per_device_train_batch_size $BS \
--save_steps 1625 --src_lang en_XX --task translation \
--label_smoothing_factor 0 --warmup_steps 500 --n_train 12895 \
--n_test 2000 --fp16 --deepspeed ds_config_stage2_gptj.json
# google/t5-v1_1-xl    BS=14
# t5-3b    BS=12
#  --dataloader_num_workers 1
# --save_steps 32500