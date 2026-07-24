"""PPOAgent 控制流与异常路径的轻量单元测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium import spaces

import src.scheduler.ppo_agent as ppo_module
from src.scheduler.ppo_agent import PPOAgent

LOG_DIR = "logs/test_ppo_agent"


class TinyEnv:
    """不触发真实训练的最小 Gym 风格环境。"""

    observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def __init__(self) -> None:
        self.steps = 0

    def reset(self):
        """开始一个两步回合。"""
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        """返回固定奖励和可预测的结束状态。"""
        del action
        self.steps += 1
        done = self.steps >= 2
        return (
            np.full(4, self.steps / 2, dtype=np.float32),
            1.5,
            done,
            False,
            {"completion_rate": 0.75 if done else 0.0},
        )


@pytest.fixture
def tiny_env() -> TinyEnv:
    """提供独立的最小环境。"""
    return TinyEnv()


def test_init_keeps_hyperparameters_and_builds_simulated_annealer(
    monkeypatch, tiny_env: TinyEnv
) -> None:
    """初始化应保存全部关键超参数并按配置创建退火器。"""
    optimizer = SimpleNamespace(simulation_mode=True)
    optimizer_cls = MagicMock(return_value=optimizer)
    monkeypatch.setattr(ppo_module, "QuantumAnnealingOptimizer", optimizer_cls)

    agent = PPOAgent(
        tiny_env,
        learning_rate=1e-4,
        n_steps=32,
        batch_size=8,
        n_epochs=3,
        gamma=0.8,
        gae_lambda=0.7,
        clip_range=0.1,
        ent_coef=0.02,
        vf_coef=0.4,
        max_grad_norm=0.3,
        seed=7,
        verbose=0,
        log_dir=LOG_DIR,
        use_annealing=True,
        anneal_interval=17,
        anneal_qubits=12,
        anneal_shots=5,
    )

    assert agent.model is None
    assert agent.learning_rate == 1e-4
    assert agent.n_steps == 32
    assert agent.batch_size == 8
    assert agent.annealing_optimizer is optimizer
    assert agent.anneal_interval == 17
    optimizer_cls.assert_called_once_with(
        num_qubits=12,
        annealing_time=20.0,
        shots=5,
        simulation_mode=True,
        cqlib_client=None,
    )


@pytest.mark.parametrize(
    ("use_lstm", "constructor_name", "policy_name"),
    [(False, "PPO", "MlpPolicy"), (True, "RecurrentPPO", "MlpLstmPolicy")],
)
def test_build_model_selects_expected_algorithm(
    monkeypatch,
    tiny_env: TinyEnv,
    use_lstm: bool,
    constructor_name: str,
    policy_name: str,
) -> None:
    """普通策略和 LSTM 策略应分别使用对应的 SB3 类。"""
    standard = MagicMock(name="PPO")
    recurrent = MagicMock(name="RecurrentPPO")
    monkeypatch.setattr(ppo_module, "PPO", standard)
    monkeypatch.setattr(ppo_module, "RecurrentPPO", recurrent)
    agent = PPOAgent(
        tiny_env,
        use_lstm=use_lstm,
        n_lstm_layers=2,
        lstm_hidden_size=24,
        log_dir=LOG_DIR,
        verbose=0,
    )

    model = agent._build_model()

    selected = recurrent if constructor_name == "RecurrentPPO" else standard
    assert model is selected.return_value
    args, kwargs = selected.call_args
    assert args[:2] == (policy_name, tiny_env)
    assert kwargs["tensorboard_log"] == LOG_DIR
    if use_lstm:
        assert kwargs["policy_kwargs"]["n_lstm_layers"] == 2
        assert kwargs["policy_kwargs"]["lstm_hidden_size"] == 24
    else:
        assert kwargs["policy_kwargs"] == {"net_arch": [128, 64]}


def _patch_training_callbacks(monkeypatch):
    """替换 SB3 回调和 Monitor，返回可断言的 Mock。"""
    eval_callback = MagicMock(name="eval_callback")
    eval_cls = MagicMock(return_value=eval_callback)
    callback_list = MagicMock(name="callback_list")
    callback_list_cls = MagicMock(return_value=callback_list)
    monkeypatch.setattr(ppo_module, "Monitor", lambda env: env)
    monkeypatch.setattr(ppo_module, "EvalCallback", eval_cls)
    monkeypatch.setattr(ppo_module, "CallbackList", callback_list_cls)
    return eval_callback, eval_cls, callback_list, callback_list_cls


def test_train_new_model_uses_single_eval_callback(monkeypatch, tiny_env: TinyEnv) -> None:
    """首次训练应构建模型，并重置时间步计数。"""
    eval_callback, _, _, callback_list_cls = _patch_training_callbacks(monkeypatch)
    model = MagicMock()
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    monkeypatch.setattr(agent, "_build_model", MagicMock(return_value=model))

    result = agent.train(total_timesteps=12, eval_freq=3, progress_bar=False)

    assert result is model
    callback_list_cls.assert_not_called()
    model.learn.assert_called_once_with(
        total_timesteps=12,
        callback=eval_callback,
        tb_log_name="ppo_scheduling",
        reset_num_timesteps=True,
        progress_bar=False,
    )


def test_train_resume_combines_annealing_real_and_extra_callbacks(
    monkeypatch, tiny_env: TinyEnv
) -> None:
    """续训应保留时间轴，并组合退火、真机和外部回调。"""
    _, _, callback_list, callback_list_cls = _patch_training_callbacks(monkeypatch)
    annealing_callback = MagicMock(name="annealing_callback")
    real_callback = MagicMock(name="real_callback")
    monkeypatch.setattr(
        ppo_module,
        "QuantumAnnealingOptimizer",
        MagicMock(return_value=SimpleNamespace(simulation_mode=True)),
    )
    annealing_cls = MagicMock(return_value=annealing_callback)
    real_cls = MagicMock(return_value=real_callback)
    monkeypatch.setattr(ppo_module, "AnnealingCallback", annealing_cls)
    monkeypatch.setattr(ppo_module, "RealMachineCallback", real_cls)
    checkpoint = "checkpoint.zip"
    monkeypatch.setattr(ppo_module.os.path, "exists", lambda path: path == checkpoint)
    model = MagicMock()
    agent = PPOAgent(
        tiny_env,
        use_annealing=True,
        anneal_interval=9,
        log_dir=LOG_DIR,
        verbose=0,
    )

    def fake_load(path: str) -> None:
        assert path == checkpoint
        agent.model = model

    monkeypatch.setattr(agent, "load", fake_load)
    extra = MagicMock(name="extra_callback")
    client = object()

    result = agent.train(
        total_timesteps=20,
        resume_from=checkpoint,
        extra_callbacks=[extra],
        real_callback_interval=4,
        real_callback_prob=0.2,
        real_callback_client=client,
        real_callback_save_path="result.json",
        real_callback_shots=64,
    )

    assert result is model
    callbacks = callback_list_cls.call_args.args[0]
    assert callbacks[1:] == [annealing_callback, real_callback, extra]
    real_cls.assert_called_once_with(
        env=tiny_env,
        interval=4,
        prob=0.2,
        client=client,
        save_path="result.json",
        shots=64,
        verbose=1,
    )
    model.learn.assert_called_once()
    assert model.learn.call_args.kwargs["callback"] is callback_list
    assert model.learn.call_args.kwargs["reset_num_timesteps"] is False


def test_predict_requires_model_and_handles_vector_shapes(
    tiny_env: TinyEnv,
) -> None:
    """推理应拒绝空模型，并把一维状态转换为批次。"""
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    with pytest.raises(RuntimeError, match="模型尚未训练"):
        agent.predict(np.zeros(4, dtype=np.float32))

    model = MagicMock()
    model.predict.return_value = (np.array([2]), None)
    agent.model = model
    assert agent.predict(np.zeros(4, dtype=np.float32), deterministic=False) == 2
    assert model.predict.call_args.args[0].shape == (1, 4)
    assert model.predict.call_args.kwargs["deterministic"] is False


def test_evaluate_aggregates_rewards_and_completion(tiny_env: TinyEnv) -> None:
    """评估应汇总每回合奖励和最终完成率。"""
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    with pytest.raises(RuntimeError, match="模型尚未训练"):
        agent.evaluate()
    model = MagicMock()
    model.predict.return_value = (np.array([1]), None)
    agent.model = model

    result = agent.evaluate(num_episodes=3, deterministic=True)

    assert result == {
        "mean_reward": 3.0,
        "std_reward": 0.0,
        "success_rate": 0.75,
        "num_episodes": 3,
    }


def test_save_load_config_and_repr(monkeypatch, tiny_env: TinyEnv) -> None:
    """模型持久化和配置展示应覆盖普通及 LSTM 加载分支。"""
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    with pytest.raises(RuntimeError, match="没有可保存"):
        agent.save("unused")

    model = MagicMock()
    agent.model = model
    agent.save("model-path")
    model.save.assert_called_once_with("model-path")

    standard_load = MagicMock(return_value=MagicMock())
    recurrent_load = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(ppo_module.PPO, "load", standard_load)
    monkeypatch.setattr(ppo_module.RecurrentPPO, "load", recurrent_load)
    agent.load("standard.zip")
    standard_load.assert_called_once_with("standard.zip", env=tiny_env)

    lstm_agent = PPOAgent(tiny_env, use_lstm=True, log_dir=LOG_DIR, verbose=0)
    lstm_agent.load("lstm.zip")
    recurrent_load.assert_called_once_with("lstm.zip", env=tiny_env)

    config = agent.get_config()
    assert config["observation_dim"] == 4
    assert config["action_dim"] == 3
    assert config["architecture"] == "PPO"
    text = repr(agent)
    assert "PPOAgent" in text
    assert "状态维度=4" in text
    assert "动作维度=3" in text


# ============================================================================
# Issue #99: 补充覆盖率 — ent_coef/clip_range/max_grad_norm 传递验证
# ============================================================================


def test_build_model_passes_all_hyperparameters(monkeypatch, tiny_env: TinyEnv) -> None:
    """_build_model 应将全部超参数（ent_coef/clip_range/max_grad_norm）传递给 PPO。"""
    standard = MagicMock(name="PPO")
    monkeypatch.setattr(ppo_module, "PPO", standard)
    agent = PPOAgent(
        tiny_env,
        learning_rate=5e-4,
        n_steps=64,
        batch_size=16,
        n_epochs=5,
        gamma=0.9,
        gae_lambda=0.8,
        clip_range=0.15,
        ent_coef=0.03,
        vf_coef=0.6,
        max_grad_norm=0.7,
        seed=42,
        log_dir=LOG_DIR,
        verbose=0,
    )

    agent._build_model()

    kwargs = standard.call_args.kwargs
    assert kwargs["ent_coef"] == 0.03
    assert kwargs["clip_range"] == 0.15
    assert kwargs["max_grad_norm"] == 0.7
    assert kwargs["learning_rate"] == 5e-4
    assert kwargs["n_steps"] == 64
    assert kwargs["batch_size"] == 16
    assert kwargs["n_epochs"] == 5
    assert kwargs["gamma"] == 0.9
    assert kwargs["gae_lambda"] == 0.8
    assert kwargs["vf_coef"] == 0.6
    assert kwargs["seed"] == 42


def test_init_annealing_real_machine_mode(monkeypatch, tiny_env: TinyEnv) -> None:
    """退火器真机模式（simulation_mode=False）应正确初始化。"""
    optimizer = SimpleNamespace(simulation_mode=False)
    optimizer_cls = MagicMock(return_value=optimizer)
    monkeypatch.setattr(ppo_module, "QuantumAnnealingOptimizer", optimizer_cls)

    agent = PPOAgent(
        tiny_env,
        log_dir=LOG_DIR,
        verbose=0,
        use_annealing=True,
        anneal_simulation_mode=False,
    )

    assert agent.annealing_optimizer is optimizer
    optimizer_cls.assert_called_once()
    assert optimizer_cls.call_args.kwargs["simulation_mode"] is False


# ============================================================================
# Issue #99: 补充覆盖率 — train 方法未覆盖分支
# ============================================================================


def test_train_with_existing_model_skips_build(monkeypatch, tiny_env: TinyEnv) -> None:
    """已有模型时应跳过 _build_model，直接进入训练（连续更新场景）。"""
    _patch_training_callbacks(monkeypatch)
    model = MagicMock()
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    agent.model = model
    build_mock = MagicMock()
    monkeypatch.setattr(agent, "_build_model", build_mock)

    result = agent.train(total_timesteps=10, eval_freq=2, progress_bar=False)

    assert result is model
    build_mock.assert_not_called()
    model.learn.assert_called_once()
    assert model.learn.call_args.kwargs["reset_num_timesteps"] is True


def test_train_with_log_dir_override(monkeypatch, tiny_env: TinyEnv) -> None:
    """train 的 log_dir 参数应覆盖默认 tb_log_name。"""
    _patch_training_callbacks(monkeypatch)
    model = MagicMock()
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    monkeypatch.setattr(agent, "_build_model", MagicMock(return_value=model))

    agent.train(total_timesteps=10, eval_freq=2, log_dir="custom_tb_log", progress_bar=False)

    assert model.learn.call_args.kwargs["tb_log_name"] == "custom_tb_log"


def test_train_resume_file_not_found_builds_new(monkeypatch, tiny_env: TinyEnv) -> None:
    """resume_from 指定的文件不存在时应回退到构建新模型。"""
    _patch_training_callbacks(monkeypatch)
    model = MagicMock()
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    build_mock = MagicMock(return_value=model)
    monkeypatch.setattr(agent, "_build_model", build_mock)
    monkeypatch.setattr(ppo_module.os.path, "exists", lambda path: False)

    agent.train(total_timesteps=10, eval_freq=2, resume_from="nonexistent.zip", progress_bar=False)

    build_mock.assert_called_once()
    assert model.learn.call_args.kwargs["reset_num_timesteps"] is True


def test_train_with_only_extra_callbacks(monkeypatch, tiny_env: TinyEnv) -> None:
    """仅有 extra_callbacks（无退火、无真机）时应使用 CallbackList 组合回调。"""
    _, _, _, callback_list_cls = _patch_training_callbacks(monkeypatch)
    model = MagicMock()
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    monkeypatch.setattr(agent, "_build_model", MagicMock(return_value=model))

    extra = MagicMock(name="extra_callback")
    agent.train(total_timesteps=10, eval_freq=2, extra_callbacks=[extra], progress_bar=False)

    callback_list_cls.assert_called_once()
    callbacks = callback_list_cls.call_args.args[0]
    assert len(callbacks) == 2
    assert callbacks[1] is extra


def test_train_real_callback_without_annealing(monkeypatch, tiny_env: TinyEnv) -> None:
    """真机回调在未启用退火时应独立工作（callbacks = [eval, real]）。"""
    _, _, _, callback_list_cls = _patch_training_callbacks(monkeypatch)
    real_callback = MagicMock(name="real_callback")
    real_cls = MagicMock(return_value=real_callback)
    monkeypatch.setattr(ppo_module, "RealMachineCallback", real_cls)
    model = MagicMock()
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    monkeypatch.setattr(agent, "_build_model", MagicMock(return_value=model))

    client = object()
    agent.train(
        total_timesteps=10,
        eval_freq=2,
        real_callback_interval=5,
        real_callback_prob=0.1,
        real_callback_client=client,
        real_callback_save_path="real.json",
        real_callback_shots=128,
        progress_bar=False,
    )

    real_cls.assert_called_once_with(
        env=tiny_env,
        interval=5,
        prob=0.1,
        client=client,
        save_path="real.json",
        shots=128,
        verbose=1,
    )
    callbacks = callback_list_cls.call_args.args[0]
    assert len(callbacks) == 2
    assert callbacks[1] is real_callback


# ============================================================================
# Issue #99: 补充覆盖率 — predict / evaluate / get_config 未覆盖分支
# ============================================================================


def test_predict_handles_2d_state(tiny_env: TinyEnv) -> None:
    """predict 应正确处理二维状态输入（不 reshape）。"""
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    model = MagicMock()
    model.predict.return_value = (np.array([1]), None)
    agent.model = model

    state_2d = np.zeros((1, 4), dtype=np.float32)
    result = agent.predict(state_2d, deterministic=True)

    assert result == 1
    assert model.predict.call_args.args[0].shape == (1, 4)


def test_evaluate_non_deterministic(tiny_env: TinyEnv) -> None:
    """evaluate 在 deterministic=False 时应将参数传递给 predict。"""
    agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
    model = MagicMock()
    model.predict.return_value = (np.array([1]), None)
    agent.model = model

    result = agent.evaluate(num_episodes=2, deterministic=False)

    assert result["mean_reward"] == 3.0
    assert result["num_episodes"] == 2
    assert model.predict.call_args.kwargs["deterministic"] is False


def test_get_config_returns_all_fields(tiny_env: TinyEnv) -> None:
    """get_config 应返回全部超参数字段。"""
    agent = PPOAgent(
        tiny_env,
        learning_rate=1e-4,
        n_steps=32,
        batch_size=8,
        n_epochs=3,
        gamma=0.8,
        gae_lambda=0.7,
        clip_range=0.1,
        ent_coef=0.02,
        vf_coef=0.4,
        max_grad_norm=0.3,
        log_dir=LOG_DIR,
        verbose=0,
    )

    config = agent.get_config()
    assert config["learning_rate"] == 1e-4
    assert config["n_steps"] == 32
    assert config["batch_size"] == 8
    assert config["n_epochs"] == 3
    assert config["gamma"] == 0.8
    assert config["gae_lambda"] == 0.7
    assert config["clip_range"] == 0.1
    assert config["ent_coef"] == 0.02
    assert config["vf_coef"] == 0.4
    assert config["max_grad_norm"] == 0.3
    assert config["architecture"] == "PPO"
    assert config["observation_dim"] == 4
    assert config["action_dim"] == 3
