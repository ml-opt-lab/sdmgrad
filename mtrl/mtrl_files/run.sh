mkdir -p ./trainlogs

#### MT10 ####

### use an inner gradient descent to solve the optimization (sdmgrad_method=sdmgrad), the default method ###
export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt10 agent=sdmgrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=1 replay_buffer.batch_size=1280 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.sdmgrad_method=sdmgrad agent.builder.agent_cfg.sdmgrad_lmbda=0.6 > trainlogs/mt10_sdmgrad-c6e-1_sd1.log 2>&1 &

### objective sampling method, sdmgrad_method=sdmgrad_osk, where k is the number of samples ###
export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt10 agent=sdmgrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=1 replay_buffer.batch_size=1280 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.sdmgrad_method=sdmgrad_os4 agent.builder.agent_cfg.sdmgrad_lmbda=0.6 > trainlogs/mt10_sdmgrad_os4-c6e-1_sd1.log 2>&1 &


#### MT50 ####

### use an inner gradient descent to solve the optimization (sdmgrad_method=sdmgrad), the default method ###
export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt50 agent=sdmgrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=1 replay_buffer.batch_size=1280 agent.multitask.num_envs=50 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.sdmgrad_method=sdmgrad agent.builder.agent_cfg.sdmgrad_lmbda=0.6 > trainlogs/mt50_sdmgrad-c6e-1_sd1.log 2>&1 &

### fast method, sdmgrad_method=sdmgrad_osk, where k is the number of gradients we calculate ###
export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt10 agent=sdmgrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=1 replay_buffer.batch_size=1280 agent.multitask.num_envs=50 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.sdmgrad_method=sdmgrad_os8 agent.builder.agent_cfg.sdmgrad_lmbda=0.6 > trainlogs/mt50_sdmgrad_os8-c6e-1_sd1.log 2>&1 &
