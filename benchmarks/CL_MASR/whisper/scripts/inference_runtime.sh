# Activate Environment
module load StdEnv/2023  gcc/12.3 intel/2023.2.1 gcccore/.12.3 ucc/1.2.0 ucx/1.14.1 openmpi/4.1.5 arrow/17.0.0 cuda/11.8
source $HOME/projects/def-ravanelm/salmanhu/benchmarks/venv/bin/activate

cd $HOME/projects/def-ravanelm/salmanhu/benchmarks/benchmarks/CL_MASR/whisper

# Bottleneck
python inference_bench_sb.py --yaml hparams/AB/decoder/RW_Bn.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json
python inference_bench_sb.py --yaml hparams/AB/encoder/RW_Bn.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json
python inference_bench_sb.py --yaml hparams/AB/enc_dec/RW_Bn.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json

# Conformer
python inference_bench_sb.py --yaml hparams/AB/decoder/RW_Con.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json
python inference_bench_sb.py --yaml hparams/AB/encoder/RW_Con.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json
python inference_bench_sb.py --yaml hparams/AB/enc_dec/RW_Con.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json

# S4A
python inference_bench_sb.py --yaml hparams/AB/decoder/S4A_Con.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json
python inference_bench_sb.py --yaml hparams/AB/encoder/S4A_Con.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json
python inference_bench_sb.py --yaml hparams/AB/enc_dec/S4A_Con.yaml --device cuda --precision bf16 --iters 30 --warmup 10 --batch-sizes 1 2 4 --seconds 5 10 20 --out runtime_results.json
