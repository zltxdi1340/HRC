HRC/
    hrc_stage1.py  #CS / IS / CCS + Update-H
    causal_discovery_simple.py  #对每个候选子节点 D，枚举 IS 的父集子集 P，找出“只有当 P 都为 1 时 D 才显著更常出现”的最小 P。
    readme.txt
    run_stage1_batch.ps1
    run_m1_experiment.ps1  
    env/
        __init__.py
        wrappers.py  # 全局禁 drop
        subgoals.py  # 抽取 subgoal 变量
        doorkey_ball_env.py  #设置新环境，图因果图
    interventions/
        __init__.py
        do_ops.py  # do-Interventions
        sampling.py  # InterventionSampling（多轨迹 + 随机顺序 + 探索窗口）
    data/

    viz/
        viz_edge_frequency.py  # 切片
    rl/
        gc_option_env.py  # 把环境包装成“目标=某个子目标节点”的 option 训练环境
        train_sb3_gc_ppo.py  # 用 SB3 的 PPO 训练（最省事）
        callbacks.py  # 加一个 SB3 Callback，按 goal 统计成功率并写入 TensorBoard
        eval_composed_task_nodrop.py
        eval_option_policy.py
        small_cnn.py




MiniGrid-Empty-8x8-v0
MiniGrid-FourRooms-v0
MiniGrid-GoToObject-8x8-N2-v0
BabyAI-UnlockLocal-v0
MiniGrid-DoorKey-6x6-v0
MiniGrid-UnlockPickup-v0
MiniGrid-Fetch-8x8-N3-v0
BabyAI-Pickup-v0
BabyAI-KeyInBox-v0
MiniGrid-KeyCorridorS3R1-v0
MiniGrid-MultiRoom-N4-S5-v0
MiniGrid-DoorKeyBall-6x6-v0


python hrc_stage1.py --env_id MiniGrid-Fetch-8x8-N3-v0 --T 100 --H 150 --delta 40 --seed 0 --out_summary outputs/stage1/fetch_summary_big.json
python hrc_stage1.py --env_id BabyAI-UnlockLocal-v0 --T 100 --H 150 --delta 40 --seed 0 --out_summary outputs/stage1/unlocklocal_summary_big.json
python hrc_stage1.py --env_id BabyAI-KeyInBox-v0 --T 100 --H 150 --delta 40 --seed 0 --out_summary outputs/stage1/keyinbox_summary_big.json
python hrc_stage1.py --env_id MiniGrid-DoorKey-6x6-v0 --T 100 --H 150 --delta 40 --seed 0 --out_summary outputs/stage1/doorkey_summary_big.json
python hrc_stage1.py --env_id MiniGrid-Empty-8x8-v0 --T 40 --H 80 --delta 20 --seed 0 --out_summary outputs/stage1/empty_summary.json
python hrc_stage1.py --env_id MiniGrid-FourRooms-v0 --T 40 --H 80 --delta 20 --seed 0 --out_summary outputs/stage1/fourrooms_summary.json
python hrc_stage1.py --env_id MiniGrid-UnlockPickup-v0 --T 40 --H 80 --delta 20 --seed 0 --out_summary outputs/stage1/unlockpickup_summary.json
python hrc_stage1.py --env_id BabyAI-Pickup-v0 --T 40 --H 80 --delta 20 --seed 0 --out_summary outputs/stage1/babyaipickup_summary.json
python hrc_stage1.py   --env_id MiniGrid-GoToObject-8x8-N2-v0   --T 40   --H 80   --delta 20   --seed 0   --out_summary outputs/stage1/gotoobject_summary.json


python viz/viz_edge_frequency.py --summary_glob "outputs/stage1/*_summary.json" --out_dir outputs/stage1
或者选取几个：
python viz/viz_edge_frequency.py --summaries `
  outputs/stage1/empty_summary.json `
  outputs/stage1/fourrooms_summary.json `
  outputs/stage1/gotoobject_summary.json `
  outputs/stage1/fetch_summary.json `
  outputs/stage1/babyaipickup_summary.json `
  outputs/stage1/doorkey_summary.json `
  outputs/stage1/unlocklocal_summary.json `
  outputs/stage1/keyinbox_summary.json `
  outputs/stage1/unlockpickup_summary.json `
  --out_dir outputs/stage1


tensorboard --logdir runs_gcppo

Windows + conda 环境
Python 3.10.19
MiniGrid 3.0.0（gymnasium 接口）


Set-ExecutionPolicy -Scope Process Bypass
.\run_m1_experiment.ps1

是的我同意你说全局统一成自动构造“与当前 locked door 匹配颜色的 key，但是我不认为keyinbox必须强制open_box=1，我们研究的是钥匙和门的关系，不是箱子是否打开和门的关系，会引入额外干扰；我还想问，如果环境存在着多个锁着的门，那又应该匹配颜色呢？目前上传给你的这个文件应该就是干预如何操作的具体文件了