# A-OkVQA
`python scripts/run_eval.py --model qwenvl --model-path "D:\Anum\VLMs\VHR-main\models\Qwen2-VL-7B-Instruct" --bench aokvqa --out results/qwenvl/aokvqa_eval.json`

# TextVQA
`python scripts/run_eval.py --model qwenvl --model-path "D:\Anum\VLMs\VHR-main\models\Qwen2-VL-7B-Instruct" --bench textvqa --questions-json ../VHR-main/data/textvqa/TextVQA_0.5.1_val.json --image-dir ../VHR-main/data/textvqa/train_images --out results/qwenvl/textvqa_eval_llava.json`

# POPE
`python scripts/run_eval.py --model instructblip --model-path D:\Anum\VLMs\VHR-main\models\instructblip-vicuna-7b --bench pope --pope-json data/pope/coco_pope_random.json --image-dir ../VHR_original/data/coco/val2014 --out results/instructblip/pope_rand.json`