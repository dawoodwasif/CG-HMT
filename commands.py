# pip install -r requirements.txt
# $env:CARLA_SERVER="G:\CARLA\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe"
# python .\CG-HMT-PPO.py --num-iters 100 --train-batch-size 12000 --rollout-fragment-length 200
# python .\CG-HMT-PPO-attacks.py --num-iters 100 --train-batch-size 12000 --rollout-fragment-length 200 --attack v2x
# python evaluate_checkpoint.py --script ".\CG-HMT-PPO.py" --checkpoint ".\ray_results\platoon_mass5_imm_trust_human_run_2025_12_21_20_10_05\MA-PPO-IMMTrust-Human-CARLA-MASS5\PPO_HomoNcomIndePOIntrxMASS3CTWN3-v0_0_2025-12-21_20-10-05pe4h84je\checkpoint_100\checkpoint-100" --seeds-file ".\eval_seeds.txt" --episodes 10 --out-prefix "cghmt_eval"
