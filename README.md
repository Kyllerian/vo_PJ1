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
| PPO | -1142.73 | n/a | 47 |
| SAC | -974.11 | n/a | 47 |

**График:** `track1/artifacts/plots/exp1_learning_curve.png`

### Эксперимент 2: PPO small vs PPO large
**Гипотеза:** более крупная сеть улучшит финальную награду, но может быть менее стабильной в начале.

**Контроль:** одинаковые timesteps (10k), один seed (47), одинаковые гиперпараметры PPO.

**Результаты (mean reward по 20 эпизодам):**
| Вариант | Mean reward | Std | Seed |
|---|---:|---:|---:|
| PPO_small | -1142.73 | n/a | 47 |
| PPO_large | -1120.19 | n/a | 47 |

**График:** `track1/artifacts/plots/exp2_learning_curve.png`

**Видео лучшего агента:** `track1/artifacts/videos/exp2_PPO_large_seed_47.mp4`

**Комментарий:** SAC показывает лучшую среднюю награду на том же бюджете. Увеличение сети PPO даёт небольшое улучшение.

## Трек 2: GridWorld (кастомная среда)
**Описание среды:** сетка 4x4, одна яма, цель в правом нижнем углу, `max_steps=30`. Наблюдение: нормированные координаты агента и цели. Два режима награды: `sparse` и `dense` (по delta расстояния до цели).

**Гипотеза:** dense reward ускорит обучение и повысит успех по сравнению со sparse.

**Контроль:** одинаковые timesteps (30k), один seed (50), одинаковая схема оценки (20 эпизодов).

**Результаты:**
| Вариант | Mean reward | Success rate | Mean length | Seed |
|---|---:|---:|---:|---:|
| dense | 0.33 | 0.0 | 30.0 | 50 |
| sparse | 0.95 | 1.0 | 6.0 | 50 |

**График:** `track2/artifacts/plots/track2_learning_curve.png`

**Видео лучшего агента:** `track2/artifacts/videos/track2_sparse_seed_50.mp4`

**Краткий анализ:** гипотеза не подтвердилась — sparse-режим привёл к стабильному достижению цели, а dense-режим за 30k шагов не добрался до успеха. Возможные причины: слишком слабое подкрепление в dense-режиме и недостаточная длительность обучения. Для улучшения можно увеличить timesteps или усилить reward shaping.

## Артефакты
- `track1/artifacts/logs/` — логи Monitor и TensorBoard
- `track1/artifacts/models/` — модели PPO/SAC
- `track1/artifacts/plots/` — графики обучения и таблицы CSV
- `track1/artifacts/videos/` — видео агента
- `track2/artifacts/...` — аналогично для GridWorld

## Воспроизводимость и допущения
- Запуски для отчёта выполнялись на CPU; по умолчанию device теперь `cuda`, поэтому при отсутствии GPU используйте `--device cpu`.
- Бюджет timesteps уменьшен для ускорения (Track1: 10k, Track2: 30k).
- Полный список пакетов: `pip_freeze.txt`.
