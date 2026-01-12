# Учебный проект по Reinforcement Learning

Проект содержит два трека:
- **Трек 1**: обучение в классической среде `Pendulum-v1` и два контролируемых эксперимента.
- **Трек 2**: собственная среда `GridWorld` с режимами награды (sparse vs dense).

Все артефакты (логи, модели, графики, видео) сохраняются в `track1/artifacts/` и `track2/artifacts/`.

## Структура
```
project_root/
  RL_Project.ipynb
  README.md
  requirements.txt
  pip_freeze.txt
  track1/
    train.py
    evaluate.py
    plot_learning_curves.py
    configs.py
    artifacts/
  track2/
    envs/gridworld.py
    train.py
    evaluate.py
    plot_learning_curves.py
    artifacts/
  utils/
    seed.py
    video.py
    plotting.py
```

## Как запускать локально
1) Установка зависимостей:
```
python -m pip install -r requirements.txt
```

По умолчанию `--device cuda`. Если CUDA недоступна, добавьте `--device cpu` к командам ниже.

2) Трек 1 (Pendulum-v1):
```
python track1/train.py --experiment exp1 --seeds 47 --timesteps 10000
python track1/train.py --experiment exp2 --seeds 47 --timesteps 10000
python track1/plot_learning_curves.py --experiment exp1 --seeds 47
python track1/plot_learning_curves.py --experiment exp2 --seeds 47
python track1/evaluate.py --experiment exp1 --seeds 47 --n_eval_episodes 20
python track1/evaluate.py --experiment exp2 --seeds 47 --n_eval_episodes 20 --record_video
```

3) Трек 2 (GridWorld):
```
python track2/train.py --variants sparse,dense --seeds 50 --timesteps 30000
python track2/plot_learning_curves.py --seeds 50 --variants sparse,dense
python track2/evaluate.py --seeds 50 --variants sparse,dense --n_eval_episodes 20 --record_video
```

4) Версии пакетов:
```
python -m pip freeze > pip_freeze.txt
```

## Как открыть в Google Colab
Откройте `RL_Project.ipynb` в Colab (загрузите папку проекта или подключите Google Drive). Ноутбук запускается сверху вниз и сохраняет артефакты в `track1/artifacts/` и `track2/artifacts/`.

## Трек 1: Pendulum-v1
### Эксперимент 1: PPO vs SAC
**Гипотеза:** SAC быстрее выходит на более высокую среднюю награду, чем PPO, т.к. он off-policy и эффективнее использует опыт.

**Контроль:** одинаковые timesteps (10k), один seed (47), одинаковая схема оценки (20 эпизодов).

**Результаты (mean reward по 20 эпизодам):**
| Вариант | Mean reward | Std | Seed |
|---|---:|---:|---:|
| PPO | -1148.04 | n/a | 47 |
| SAC | -971.82 | n/a | 47 |

**График:** `track1/artifacts/plots/exp1_learning_curve.png`

### Эксперимент 2: PPO small vs PPO large
**Гипотеза:** более крупная сеть улучшит финальную награду, но может быть менее стабильной в начале.

**Контроль:** одинаковые timesteps (10k), один seed (47), одинаковые гиперпараметры PPO.

**Параметры (отчетный прогон):**
- seed: 47; timesteps: 10k; eval: 20 эпизодов.
- PPO: lr 3e-4, gamma 0.99, n_envs 4, net_arch [64, 64], batch_size 64.
- SAC: lr 3e-4, gamma 0.99, n_envs 1, net_arch [64, 64], buffer_size 50k, batch_size 128, train_freq 4, gradient_steps 1, learning_starts 1000.
- exp2: PPO_small net_arch [64, 64]; PPO_large net_arch [256, 256] (остальные параметры как у PPO).

**Результаты (mean reward по 20 эпизодам):**
| Вариант | Mean reward | Std | Seed |
|---|---:|---:|---:|
| PPO_small | -1148.04 | n/a | 47 |
| PPO_large | -1132.96 | n/a | 47 |

**График:** `track1/artifacts/plots/exp2_learning_curve.png`

**Видео лучшего агента:** `track1/artifacts/videos/exp2_PPO_large_seed_47.mp4`

**Краткий анализ (Трек 1):** SAC показал более высокую среднюю награду, что подтверждает гипотезу про преимущество off-policy при равном бюджете (разница около 176 по mean reward). Увеличение сети PPO дало небольшой прирост, но эффект слабый на 10k шагах. Std не оценивался из-за одного сида, поэтому выводы предварительные. Для более надежных выводов стоит увеличить число сидов и timesteps.

## Трек 2: GridWorld (кастомная среда)
**Описание среды:** сетка 4x4, одна яма, цель в правом нижнем углу, `max_steps=30`. Наблюдение: нормированные координаты агента и цели. Два режима награды: `sparse` и `dense` (по delta расстояния до цели).

**Гипотеза:** dense reward ускорит обучение и повысит успех по сравнению со sparse.

**Контроль:** одинаковые timesteps (30k), один seed (50), одинаковая схема оценки (20 эпизодов).

**Параметры (отчетный прогон):**
- seed: 50; timesteps: 30k; eval: 20 эпизодов.
- DQN: lr 1e-3, gamma 0.99, buffer_size 50k, batch_size 64, exploration_fraction 0.2, exploration_final_eps 0.05, train_freq 1, target_update_interval 500, net_arch [64, 64].

**Результаты:**
| Вариант | Mean reward | Success rate | Mean length | Seed |
|---|---:|---:|---:|---:|
| dense | 1.83 | 1.0 | 6.0 | 50 |
| sparse | 0.95 | 1.0 | 6.0 | 50 |

**График:** `track2/artifacts/plots/track2_learning_curve.png`

**Видео лучшего агента:** `track2/artifacts/videos/track2_dense_seed_50.mp4`

**Краткий анализ:** по итоговой оценке обе версии стабильно достигают цель (success_rate 1.0, mean_length 6.0). Dense дает более высокий mean_reward (1.83 vs 0.95), поэтому гипотеза частично подтверждается по величине награды. Так как использован один сид, выводы ограничены; полезно увеличить timesteps и число сидов или усилить reward shaping.

## Артефакты
- `track1/artifacts/logs/` — логи Monitor и TensorBoard
- `track1/artifacts/models/` — модели PPO/SAC
- `track1/artifacts/plots/` — графики обучения и таблицы CSV
- `track1/artifacts/videos/` — видео агента
- `track2/artifacts/...` — аналогично для GridWorld

## Воспроизводимость и допущения
- Запуски для отчёта выполнялись на CPU; по умолчанию device теперь `cuda`, поэтому при отсутствии GPU используйте `--device cpu`.
- Бюджет timesteps уменьшен для ускорения (Track1: 10k, Track2: 30k).
- Отчетные результаты получены при seed=47 (Track1) и seed=50 (Track2); std_reward не рассчитывался из-за одного сида.
- Полный список пакетов: `pip_freeze.txt`.
- Полные конфиги с гиперпараметрами: `track1/configs.py` и `track2/configs.py`.
