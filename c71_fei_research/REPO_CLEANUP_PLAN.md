# План аудита и реорганизации репозитория `исследования` перед стартом трека `C_71` (FEI)

**Вердикт:** Репозиторий в чистом состоянии (`git status --short`: 0 изменений, ветка `main`, `HEAD = 477024cffe4b2e28c6b310e879e38617d969f3ef`, remote `sunil123456897/lean-dag-rsi.git`). Фактический вес рабочей копии без `.git` — **252.90 MB (422 файла: 218 tracked, 0 untracked, 204 ignored)**, из которых **91.4% (231.25 MB)** занимает локальная PDF-библиотека `Статьи_RSI_и_Самосовершенствование/` (игнорируемая git), включая **48.60 MB (32 файла) 100%-ных SHA-256 дублей** в `00_ОТОБРАННОЕ_ЯДРО_ЛУЧШИХ_ИДЕЙ/`. Все пути открытых PR (`c89_certificate_package/`, `heilbronn_pr_package/`) и ядро плагина (`dream_rsi_scripts/`, `plugin_bundle/`, `prompts/`, `sync_plugin.py`, `sleep_dag_rsi.py`, `AGENTS.md`) зафиксированы в **Tier C (неприкосновенно)**.

> **Метод и покрытие аудита (read-only):** Полный обход дерева `e:\эксперименты\исследования` (`os.walk` + `git ls-files` + побайтовый `SHA-256` всех непустых файлов вне `.git`/кэшей + статический поиск импортов и строковых ссылок по всем `.py`/`.md`/`.tex`/`.cpp`/`.lean`/`.txt`). Все числа и номера строк ниже имеют статус `VERIFIED`. Ни один файл репозитория и история `git` не изменялись.

---

## 1. Сводная инвентаризация верхнего и 2-го уровня

Всего в корне (без `.git` — `14.89 MB`, `528` файлов): **26 объектов** (`16` директорий, `10` файлов).

### 1.1. Таблица верхнего уровня

| Тип | Путь | Размер | Файлов | `tracked` | `ignored` | Назначение / статус | Tier |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :---: |
| DIR | `.ruff_cache/` | 1.7 KB | 8 | 0 | 8 | Кэш линтера Ruff (`0.14.6/`) | **A** |
| DIR | `__pycache__/` | 382.8 KB | 8 | 0 | 8 | Байткод Python 3.12 корневых скриптов | **A** |
| DIR | `из чата/` | 894.6 KB | 10 | 0 | 10 | Дамп из чата: 5 точных SHA-256 дублей + 3 устаревших v5-файла + `__pycache__` | **A** |
| DIR | `cpp/` | 3.02 MB | 11 | 4 | 7 | 4 исходника `.cpp` (137.5 KB, tracked) + 7 бинарников `.exe`/`.obj` (2.89 MB, ignored) | **A / B** |
| DIR | `Статьи_RSI_и_Самосовершенствование/` | 231.25 MB | 138 | 0 | 138 | Локальная библиотека PDF/MD; `00_ОТОБРАННОЕ_ЯДРО...` (48.60 MB, 32 файла) — 100% дубли | **A / B** |
| DIR | `artifacts/` | 554.1 KB | 27 | 27 | 0 | Чемпионы `.npy`/`.json` (19 используются кодом, 8 не ссылаются из кода) | **B** |
| DIR | `benchmarks/` | 415.4 KB | 10 | 5 | 5 | 4 бенчмарка + `__init__.py` (188.1 KB) + `__pycache__/` (227.3 KB, Tier A) | **B** |
| DIR | `btc_sleep_dag_rsi/` | 7.62 MB | 13 | 0 | 13 | Локальная трейдинг-песочница BTCUSDT (`data/*.parquet` = 7.13 MB, gitignored) | **B** |
| DIR | `docs/` | 5.63 MB | 17 | 17 | 0 | Препринты v8 (RU/EN), `concepts_brief.md` + архивные v1–v5 черновики/PDF | **B** |
| FILE | `auto_dream_rsi.py` | 106.7 KB | 1 | 1 | 0 | Harness 8 картриджей; импортирует `benchmarks.benchmark_world_records`, `verify_extremal_cartridges_7_8` | **B** |
| FILE | `lean_dag_engine.py` | 29.6 KB | 1 | 1 | 0 | Устаревший автономный скрипт v2 (0 импортов из кода; 3 ссылки в старых `docs/`) | **B** |
| FILE | `verify_dream_rsi.py` | 40.2 KB | 1 | 1 | 0 | Аудит Dream-RSI Eq.(1) + Lasso + `cpp/benchmark_lasso_cpp.exe` (0 импортов из кода) | **B** |
| FILE | `verify_extremal_cartridges_7_8.py` | 32.0 KB | 1 | 1 | 0 | Верификатор `C_1`, `C_2`, `Phi_3`, Heilbronn 13/14 (импортируется в `auto_dream_rsi.py`) | **B** |
| FILE | `verify_math_tasks.py` | 14.4 KB | 1 | 1 | 0 | Автономный верификатор математических задач v2 (0 импортов из кода) | **B** |
| DIR | `c89_certificate_package/` | 805.9 KB | 6 | 5 | 1 | Сертификат $C_{89} \le 0.0301388757$ (открытый PR `teorth/optimizationproblems#380`) + `__pycache__` (Tier A) | **C** |
| DIR | `heilbronn_pr_package/` | 567.6 KB | 121 | 121 | 0 | PR `#148`/`#151` ($n=13..17$) + staged mega-PR 55 рекордов (`triangle_world_records/`, 113 файлов) | **C** |
| DIR | `c1d_certificate_package/` | 109.4 KB | 4 | 3 | 1 | Сертификат $C_{1d} \le 1.4569611398$ ($P=6400$) + `__pycache__` (Tier A) | **C** |
| DIR | `lean4_certificates/` | 5.2 KB | 2 | 2 | 0 | Формальные сертификаты Lean 4 (`Defs.lean`, `Cert_Heilbronn_13_14.lean`) | **C** |
| DIR | `dream_rsi_scripts/` | 1.40 MB | 27 | 14 | 13 | Ядро `SLEEP-DAG-RSI v8` (`engine.py`, `sleep_dag_rsi.py`, `_selftest.py`, etc.) + `__pycache__` (Tier A) | **C** |
| DIR | `plugin_bundle/` | 28.9 KB | 9 | 9 | 0 | Источник плагина `autonomous-research v2.0` (`plugin.json`, 3 skills, 3 refs, 2 agents) | **C** |
| DIR | `prompts/` | 3.6 KB | 1 | 1 | 0 | `policy_improvement.md` — **синхронизируется `sync_plugin.py:140` и используется `loop.py:308`** | **C** |
| FILE | `sync_plugin.py` | 11.2 KB | 1 | 1 | 0 | Атомарный синхронизатор и SHA-256 валидатор плагина (`--verify`) | **C** |
| FILE | `sleep_dag_rsi.py` | 597 B | 1 | 1 | 0 | Тонкий фасад для `dream_rsi_scripts/sleep_dag_rsi.py` (проверяется `sync_plugin.py:255-259`) | **C** |
| FILE | `AGENTS.md` | 8.5 KB | 1 | 1 | 0 | Контракт и указатели контекста для агентов | **C** |
| FILE | `README.md` | 64.9 KB | 1 | 1 | 0 | Главная витрина GitHub-репозитория | **C** |
| FILE | `.gitignore` | 449 B | 1 | 1 | 0 | Правила исключений git (в Tier A предложены точечные дополнения) | **C** |

### 1.2. Ключевые находки 2-го уровня и скрытые расхождения

1. **Незадокументированная связность `prompts/` и `sleep_dag_rsi.py` с ядром плагина:**
   - [sync_plugin.py](file:///e:/эксперименты/исследования/sync_plugin.py#L4-L6) и [L139-L148](file:///e:/эксперименты/исследования/sync_plugin.py#L139-L148) жёстко синхронизируют `prompts/policy_improvement.md`, а [dream_rsi_scripts/loop.py](file:///e:/эксперименты/исследования/dream_rsi_scripts/loop.py#L308) задаёт `DEFAULT_PROMPT = "prompts/policy_improvement.md"`. Переносить `prompts/` нельзя.
   - [sync_plugin.py](file:///e:/эксперименты/исследования/sync_plugin.py#L255-L259) проверяет размер корневого `sleep_dag_rsi.py` (`<= 2048 B`), а [benchmarks/battle_test_sleep_dag_v6.py](file:///e:/эксперименты/исследования/benchmarks/battle_test_sleep_dag_v6.py) и [benchmarks/run_sleep_dag_experiments.py](file:///e:/эксперименты/исследования/benchmarks/run_sleep_dag_experiments.py) импортируют `sleep_dag_rsi` напрямую.
2. **Полный аудит дублей (SHA-256):**
   - `Статьи_RSI_и_Самосовершенствование/00_ОТОБРАННОЕ_ЯДРО_ЛУЧШИХ_ИДЕЙ/` содержит **32 файла (`48.60 MB`)**, и **все 32 из 32** являются побайтовыми копиями файлов из подпапок `01_...`–`05_...` (**0 уникальных файлов**).
   - `из чата/` (`894.6 KB`, 10 файлов, gitignored, 0 ссылок в репозитории):
     - **5 файлов** — точные SHA-256 дубли (`download.pdf` = `docs/lean_dag_rsi_v5_sleep_paper.pdf`; `n17_exact_500dps.json` = `heilbronn_pr_package/n17_exact_500dps.json`; `lean_dag_rsi_v5_sleep.md` и обе копии `sleep_dag_rsi_unification_report*.md` совпадают с копиями в `Статьи_RSI.../05_...`).
     - **3 файла** — устаревшие v5-версии уже развитых файлов: `sleep_dag_rsi.py` (`1454` строк против `3963` в `dream_rsi_scripts/sleep_dag_rsi.py`), `run_sleep_dag_experiments.py` и его копия `run_sleep_dag_experiments (1).py` (`1086` строк против `1488` в `benchmarks/run_sleep_dag_experiments.py`).
     - **2 файла** — `__pycache__/*.pyc` (`138.7 KB`).
3. **Разбор `artifacts/` (27 файлов, `554.1 KB`):**
   - **19 файлов** активно загружаются или создаются скриптами (`auto_dream_rsi.py`, `verify_extremal_cartridges_7_8.py`, `benchmarks/battle_test_sleep_dag_v6.py`, `benchmarks/storm_heilbronn_n16.py`, `cpp/heilbronn_*.cpp`).
   - **1 файл** (`phi3_p6400_champion.npy`, `50.1 KB`) упоминается в `AGENTS.md` и `README.md` как `.npy`-источник рекорда `C_1d` ($P=6400$).
   - **7–8 промежуточных/неиспользуемых кодом `.npy`** (`autocorr_c1_b1_p20.npy`, `autocorr_c1_b2_p40.npy`, `autocorr_c1_xover_p320.npy`, `heilbronn_n11_b0.npy`, `heilbronn_n11_b1.npy`, `heilbronn_n15_candidate.npy`, `heilbronn_n16_candidate.npy`, `heilbronn_n17_candidate.npy` — суммарно `5.2 KB`): ни один скрипт их не читает.
   - Примечание: [c1d_certificate_package/SUBMISSION.md](file:///e:/эксперименты/исследования/c1d_certificate_package/SUBMISSION.md#L41) ссылается в таблице на путь `artifacts/phi3_best_candidate.npy`.
4. **Дрейф документации (non-blocking finding):**
   - Ни `README.md`, ни `AGENTS.md` пока не упоминают добавленные в последних коммитах `c89_certificate_package/` (коммиты `e51392a`, `477024c`) и `heilbronn_pr_package/triangle_world_records/` (55 рекордов, коммит `d074e6d`). Стоит дописать при следующем плановом обновлении документации.

---

## 2. Трёхуровневый план уборки (Tier A / Tier B / Tier C)

### Tier A — Гигиена с нулевым риском (кэши, временные файлы, `.gitignore`)

Не затрагивает ни одного `git`-отслеживаемого файла и не ломает ни один публичный URL / PR.

| # | Объект | Объём / файлы | Статус git | Действие | Примечание / проверка |
| :---: | :--- | ---: | :---: | :--- | :--- |
| A1 | Все 8 кэш-директорий (`.ruff_cache/`, корневой и вложенные `__pycache__/`) | `1.61 MB` (40) | `ignored` | Удалить | Воссоздаются автоматически при запуске Python/Ruff |
| A2 | `из чата/` (целиком) | `894.6 KB` (10) | `ignored` | Удалить (или вынести за пределы репо) | 0 ссылок в коде; 5 точных дублей + 3 старых v5-черновика + `__pycache__` |
| A3 | `cpp/*.exe`, `cpp/*.obj` (7 бинарников) | `2.89 MB` (7) | `ignored` | Удалить (опционально оставить `benchmark_lasso_cpp.exe` если запускаете `verify_dream_rsi.py`) | [verify_dream_rsi.py:713](file:///e:/эксперименты/исследования/verify_dream_rsi.py#L713) ищет `cpp/benchmark_lasso_cpp.exe` (при отсутствии пересобирается из `.cpp`) |
| A4 | `Статьи_RSI_и_Самосовершенствование/00_ОТОБРАННОЕ_ЯДРО_ЛУЧШИХ_ИДЕЙ/` | `48.60 MB` (32) | `ignored` | Опционально дедуплицировать (заменить на индекс или удалить копии) | Все 32 файла имеют 100% SHA-256 совпадения в подпапках `01..05`; `ОТБОР_ПО_КАЧЕСТВУ_И_ИДЕЯМ.md` сохраняет список |
| A5 | Обновление `.gitignore` | `+12` строк | `tracked` | Добавить паттерны для временных артефактов и тяжёлых бинарных дампов нового трека `c71_fei_research/` | Защита от случайного коммита `.pt`/`.ckpt`/`.log`/scratch-таблиц истинности ($n \ge 20$) |

**Итого экономия на диске от Tier A:**
- Базовая уборка (`A1 + A2 + A3`, без двойного счёта `из чата/__pycache__`): **`5.24 MB` (55 файлов)**.
- С дедупликацией `00_ОТОБРАННОЕ_ЯДРО_ЛУЧШИХ_ИДЕЙ/` (`A1..A4`): **`53.83 MB` (87 файлов)**.

**Рекомендуемые дополнения в `.gitignore` (шаг A5):**
```gitignore
# Experiment logs, checkpoints & temporary scratch outputs
*.log
*.tmp
*.bak
*.pt
*.pth
*.ckpt

# C_71 FEI heavy raw search dumps (keep compact verified certificates tracked)
c71_fei_research/artifacts/raw_runs/
c71_fei_research/artifacts/checkpoints/
```

**Команды выполнения Tier A (PowerShell, запускать ТОЛЬКО после подтверждения пользователя):**
```powershell
# A1: Очистка всех __pycache__ и .ruff_cache внутри репозитория
Get-ChildItem -LiteralPath "e:\эксперименты\исследования" -Directory -Recurse -Force -Include "__pycache__",".ruff_cache" |
  Where-Object { $_.FullName -notlike "*\.git\*" } |
  Remove-Item -Recurse -Force

# A2: Удаление локального дампа `из чата/` (gitignored)
Remove-Item -LiteralPath "e:\эксперименты\исследования\из чата" -Recurse -Force

# A3: Удаление скомпилированных бинарников в cpp/ (исходники .cpp остаются нетронутыми)
Get-ChildItem -LiteralPath "e:\эксперименты\исследования\cpp" -File -Include "*.exe","*.obj" |
  Remove-Item -Force
```

---

### Tier B — Структурное упорядочивание (завершённые эксперименты и локальные тяжеловесы)

Цель Tier B — разгрузить корень репозитория и локальный диск, не ломая пути активных PR (Tier C). Рекомендуется выполнять пошагово, начиная с **B1–B2 (zero-git-impact)** и переходя к **B3–B5** только если нужна чистая витрина корня.

| # | Объект | Объём / файлы | Рекомендованное действие | Затронутые импорты / ссылки, которые нужно обновить |
| :---: | :--- | ---: | :--- | :--- |
| **B1** | `Статьи_RSI_и_Самосовершенствование/` | `231.25 MB` (138, `ignored`) | **Вынести за пределы git-репозитория** (например, в `e:\эксперименты\Библиотека_RSI\`). Оставить запись в `.gitignore` или убрать после переноса. | **0 ссылок в коде и README.** Мгновенно снижает размер рабочей директории с `252.90 MB` до `~21.65 MB`. |
| **B2** | `btc_sleep_dag_rsi/` | `7.62 MB` (13, `ignored`) | Вынести в соседний проект (`e:\эксперименты\btc_sleep_dag_rsi\`) или оставить в `experiments/btc_sleep_dag_rsi/` (добавив путь в `.gitignore`). | 1 ссылка в [AGENTS.md](file:///e:/эксперименты/исследования/AGENTS.md) (`btc_sleep_dag_rsi/`). Импорты внутри автономны. |
| **B3** | `docs/` (устаревшие v1–v5 черновики и сырые логи) | `5.63 MB` (17, `tracked`) | Оставить в `docs/` актуальное ядро (`concepts_brief.md`, `SLEEP_DAG_RSI_v8_Preprint_EN.{md,pdf,tex}`, `Полная_Статья_SLEEP_DAG_RSI_v8_RU.{md,pdf}`), а 10 архивных v1–v5 файлов (`Исследования.txt`, `dream-rsi.pdf`, `Исследование_Dream_RSI_*.md`, `Полная_Статья_Lean_DAG_RSI_v3.{md,pdf}`, `lean_dag_rsi_v3_paper.tex`, `Полная_Статья_Lean_DAG_RSI_v5_SLEEP.md`, `lean_dag_rsi_v5_sleep_paper.pdf`, `sleep_dag_rsi_unification_report.md`) перенести в `docs/archive/` через `git mv`. | Обновить ссылки в `README.md` (если указывают на v3/v5 статьи). Код на `docs/` не ссылается. |
| **B4** | Устаревшие автономные корневые скрипты (`lean_dag_engine.py`, `verify_math_tasks.py`, `verify_dream_rsi.py`) + `cpp/*.cpp` | `221.7 KB` (7, `tracked`) | Перенести через `git mv` в `experiments/legacy_v1_v3/` (или `archive/legacy_v1_v3/`), **кроме** случая если планируется запускать `verify_dream_rsi.py` без правки пути к `cpp/`. | • `lean_dag_engine.py`: 0 импортов из кода (только 3 старые статьи в `docs/`).<br>• `verify_math_tasks.py`: 0 импортов из кода; обновить упоминание в `AGENTS.md`.<br>• `verify_dream_rsi.py`: обновить путь `cpp` в [L713](file:///e:/эксперименты/исследования/verify_dream_rsi.py#L713), `AGENTS.md` и `README.md`.<br>• `cpp/`: обновить `README.md`. |
| **B5** | Активный пакет экспериментов v4–v7 (`auto_dream_rsi.py`, `verify_extremal_cartridges_7_8.py`, `benchmarks/`, `artifacts/`) | `1.11 MB` (34, `tracked`) | **Два варианта на выбор:**<br>• **Вариант B5-Light (Рекомендуется):** оставить на месте до закрытия всех PR по Heilbronn/$C_{1d}$, только почистить 8 неиспользуемых промежуточных `.npy` в `artifacts/` (или сложить их в `artifacts/intermediate/`).<br>• **Вариант B5-Move:** перенести в `experiments/extremal_v4_v7/` целиком (сохраняя `artifacts/` рядом или прописав общий `REPO_ROOT`). | При **B5-Move** потребуется правка:<br>• `auto_dream_rsi.py` -> `from benchmarks.benchmark_world_records import ...` и `from verify_extremal_cartridges_7_8 import ...` + `artifacts_dir`.<br>• `verify_extremal_cartridges_7_8.py:498` -> путь к `lean4_certificates/` и `artifacts/`.<br>• `benchmarks/battle_test_sleep_dag_v6.py` и `benchmarks/run_sleep_dag_experiments.py` -> `ROOT_DIR`, импорты `sleep_dag_rsi` / `auto_dream_rsi`, пути к `artifacts/` и `heilbronn_pr_package/`.<br>• [c1d_certificate_package/SUBMISSION.md:41](file:///e:/эксперименты/исследования/c1d_certificate_package/SUBMISSION.md#L41) ссылается на `artifacts/phi3_best_candidate.npy`.<br>• `AGENTS.md` и `README.md` (команды верификации). |

> **Почему B5-Light предпочтительнее прямо сейчас:** `auto_dream_rsi.py`, `verify_extremal_cartridges_7_8.py`, `benchmarks/` и `artifacts/` весят всего `1.11 MB`, связаны перекрёстными импортами и завязаны на команды верификации из `AGENTS.md` и `README.md`. Вынос `Статьи_RSI_и_Самосовершенствование/` (B1) + `btc_sleep_dag_rsi/` (B2) + Tier A уже решает >95% захламлённости без единой правки Python-импортов.

---

### Tier C — Защищённая зона (STRICT DO-NOT-TOUCH)

Ни один файл и ни один путь из таблицы ниже **нельзя переименовывать, перемещать, удалять или переписывать в истории git** (`rebase`/`filter-repo` запрещены):

| Путь | Причина строгой защиты | Контрольная проверка целостности |
| :--- | :--- | :--- |
| `c89_certificate_package/` | Привязан по commit SHA (`e51392a`, `477024c`) к открытому внешнему PR **`teorth/optimizationproblems#380`**. Любое перемещение сломает ссылки в PR. | `git diff -- c89_certificate_package` пуст |
| `heilbronn_pr_package/` | Привязан к **`tejstead/heilbronn-site#148`**, **`#151`** и подготовленному mega-PR на 55 рекордов (`triangle_world_records/`). | `python heilbronn_pr_package/verify_pr_package.py` + `verify_records.py` |
| `c1d_certificate_package/` | Самодостаточный пакет сертификата $C_{1d} \le 1.4569611398$ ($P=6400$) с zero-dep brute-force верификатором. | `python c1d_certificate_package/verify_c1d.py` |
| `lean4_certificates/` | Формальные рациональные сертификаты Lean 4 (`Defs.lean`, `Cert_Heilbronn_13_14.lean`), проверяемые `verify_extremal_cartridges_7_8.py:498`. | Проверка наличия обоих `.lean` файлов |
| `dream_rsi_scripts/` | Каноническое ядро `SLEEP-DAG-RSI v8` (14 tracked-модулей, Suites 1–21). | `python dream_rsi_scripts/_selftest.py` (`RESULT: all checks passed`) |
| `plugin_bundle/` | Единый источник истины для плагина `autonomous-research v2.0` (`plugin.json`, 3 skills, 3 refs, 2 agents). | `python sync_plugin.py --verify` |
| `prompts/` | Содержит `policy_improvement.md` — **обязательный источник** для `sync_plugin.py:140` и `dream_rsi_scripts/loop.py:308`. | `python sync_plugin.py --verify` |
| `sync_plugin.py` | Атомарный синхронизатор и QA-валидатор плагина (ожидает `plugin_bundle/`, `dream_rsi_scripts/`, `prompts/` и `sleep_dag_rsi.py` в корне). | `python sync_plugin.py --verify` |
| `sleep_dag_rsi.py` | Тонкий корневой фасад (`597 B`) на `dream_rsi_scripts/sleep_dag_rsi.py`; валидируется `sync_plugin.py:255-259`. | `python sync_plugin.py --verify` |
| `AGENTS.md`, `README.md`, `.gitignore` | Корневые файлы контракта, документации и git-правил (допустимы только осознанные текстовые дополнения, не перенос). | `git status` |

---

## 3. Предлагаемая структура рабочей директории `c71_fei_research/`

Новый исследовательский трек по константе Фурье энтропия–влияние **$C_{71}$ (Fourier Entropy–Influence / Friedgut–Kalai conjecture)** изолируется в собственной самодостаточной директории `c71_fei_research/` в корне репозитория, чтобы:
1. Не смешивать новые скрипты, теории и артефакты с предыдущими треками (`Heilbronn`, `C_1`, `C_2`, `C_1d`, `C_89`).
2. Обеспечить чистую подготовку итогового PR-пакета (по образцу `c89_certificate_package/` и `c1d_certificate_package/`) без мусорных промежуточных дампов в `git`.

```text
c71_fei_research/
├── REPO_CLEANUP_PLAN.md              # Этот аудит и план реорганизации репозитория
├── README.md                         # Краткий дашборд трека C_71: цель, текущий рекорд, команды запуска
├── theory/                           # Теоретическая база и дизайн атак (только Markdown/TeX)
│   ├── 01_theory_and_sota.md         # Точные определения, теоремы O'Donnell–Tan / Hod, SOTA 2026 (> 6.521845...)
│   └── 02_constructions_and_attack_vectors.md  # Векторы атаки, симметрии, GPU FWHT + SLEEP-DAG-RSI roadmap
├── src/                              # Воспроизводимый код поиска, GPU-ядер и интеграции с SLEEP-DAG-RSI
│   ├── __init__.py
│   ├── fwht_gpu.py                   # Батчевый Walsh–Hadamard Transform на PyTorch CUDA (RTX 5060)
│   ├── fei_metrics.py                # Расчёт H[f], Inf[f], формул композиции (O'Donnell–Tan / Hod)
│   ├── constructions.py              # Параметризованные семейства (орбиты групп, блок-симметрия, немонотонные)
│   ├── cartridge_c71.py              # TaskCartridge для SLEEP-DAG-RSI / DiscoveryTree (импорт из dream_rsi_scripts)
│   └── run_search.py                 # CLI-точка входа для фаз поиска (Tracer #1 -> Baselines -> Search)
├── artifacts/                        # Результаты прогонов (лёгкие JSON-сводки tracked; тяжёлые дампы gitignored)
│   ├── baselines_summary.json        # Воспроизведённые бейзлайны и калибровка формул
│   ├── champions/                    # Компактные лучшие найденные конструкции (JSON/NPZ)
│   ├── raw_runs/                     # [gitignored] Сырые логи, таблицы истинности промежуточных прогонов
│   └── checkpoints/                  # [gitignored] Состояния DiscoveryTree / GPU-чекпоинты
└── certificates/                     # Чистые независимые пакеты сертификации (кандидаты на PR в optimizationproblems)
    ├── verify_c71_exact.py           # Zero-gaming точный верификатор (целочисленный FWHT + интервальная/mpmath арифметика)
    ├── README.md                     # Методика, SHA-256, воспроизведение и строгий вывод нижней границы C_71
    └── SHA256SUMS.txt                # Контрольные суммы финальных сертифицированных таблиц истинности
```

### Принципы гигиены для `c71_fei_research/` с первого дня
- **Никаких новых файлов в корне репозитория:** все скрипты `C_71` живут внутри `c71_fei_research/src/` и подключают `dream_rsi_scripts` через `sys.path` / пакетный импорт.
- **Разделение `artifacts/` и `certificates/`:** в `certificates/` попадают **только** окончательно проверенные рекорды с независимым exact-верификатором (по стандарту `c89_certificate_package/` и `c1d_certificate_package/`), готовые к фиксации по commit SHA в будущем PR.
- **Тяжёлые промежуточные файлы (`raw_runs/`, `checkpoints/`, `.pt`) сразу в `.gitignore`:** чтобы `.git/` оставался компактным (`< 20 MB`).

---

## 4. Рекомендуемый порядок действий для пользователя

1. **Проверить Tier A + B1** (нулевой риск для `git` и PR): очистить 8 кэшей (`1.61 MB`), удалить `из чата/` (`894.6 KB`), перенести `Статьи_RSI_и_Самосовершенствование/` (`231.25 MB`) в отдельную локальную папку вне репозитория, добавить защитные строки в `.gitignore`.
2. **Определиться по Tier B2–B5:**
   - Выносить ли `btc_sleep_dag_rsi/` (`7.62 MB`, gitignored) наружу?
   - Нужен ли сейчас перенос архивных `docs/` (B3) и legacy-скриптов v1–v3 (B4), или оставить B5-Light без рефакторинга импортов до закрытия PR по Heilbronn и $C_{89}$?
3. **После любых правок прогнать обязательный контрольный набор Tier C:**
   ```powershell
   python sync_plugin.py --verify
   python dream_rsi_scripts/_selftest.py
   python heilbronn_pr_package/verify_pr_package.py
   python c1d_certificate_package/verify_c1d.py
   git status --short
   ```
