# Досье SOTA по быстрому умножению матриц (FMM) и рейтинг атакуемых мировых рекордов (ноябрь 2026)

> **Статус сверки:** `VERIFIED` по живым первоисточникам на **2026-11-22**.
> Ни одно число не взято «из памяти» 2022–2024 годов: все таблицы перекрёстно сверены с:
> 1. **Université de Lille FMM Database** ([fmm.univ-lille.fr](https://fmm.univ-lille.fr/), 5 426 схем) и официальным дайджестом [`sedoglavic/fmm_digest/fmm_sota.json`](https://github.com/sedoglavic/fmm_digest/blob/master/fmm_sota.json) (`createdAt: 2026-11-15`, 5 456 записей);
> 2. **Каталогом А. И. Перминова** [`dronperminov/FastMatrixMultiplication`](https://github.com/dronperminov/FastMatrixMultiplication) (`schemes/status.json`, `docs/ranks.md`, `CHANGELOG.md`, `pushed: 2026-11-16`, 680 форматов до $16^3$);
> 3. **Унифицированным мультиполевым каталогом Б. Шатена Ласеля** [`solven-eu/matmulcatalog`](https://github.com/solven-eu/matmulcatalog) (`arXiv:2606.13408v3`, `docs/catalog.json`, `docs/lower-bounds.json`, `pushed: 2026-11-16`, 11 363 схемы по полям $\mathbb{F}_2, \mathbb{F}_3, \mathbb{Z}_T, \mathbb{Z}, \mathbb{Q}, \mathbb{R}, \mathbb{C}$);
> 4. **Репозиториями ноябрьских рекордов 2026:** [`gsidebottom/mat_mul`](https://github.com/gsidebottom/mat_mul) (`pushed: 2026-11-21`), [`arhancanli/quantavane-fmm-records`](https://github.com/arhancanli/quantavane-fmm-records) (`pushed: 2026-11-21`), [`khoruzhii/lita`](https://github.com/khoruzhii/lita) (`pushed: 2026-10-07`), [`mkauers/matrix-multiplication`](https://github.com/mkauers/matrix-multiplication) (`pushed: 2026-05-29`), [`big-brain-zaru/FastMatrixF2`](https://github.com/big-brain-zaru/FastMatrixF2) (`pushed: 2026-09-18`);
> 5. **Ключевыми препринтами 2024–2026 на arXiv:** Arai–Ichikawa–Hukushima ([`arXiv:2312.16960`](https://arxiv.org/abs/2312.16960)), Moosbauer–Poole ([`arXiv:2502.04514`](https://arxiv.org/abs/2502.04514)), AlphaEvolve ([`arXiv:2506.13131`](https://arxiv.org/abs/2506.13131)), Dumas–Pernet–Sedoglavic ([`arXiv:2506.13242v7`](https://arxiv.org/abs/2506.13242), [`arXiv:2603.18699v3`](https://arxiv.org/abs/2603.18699), [`arXiv:2609.12027v2`](https://arxiv.org/abs/2609.12027)), Kauers–Wood ([`arXiv:2510.19787`](https://arxiv.org/abs/2510.19787)), Perminov ([`arXiv:2511.20317`](https://arxiv.org/abs/2511.20317), [`arXiv:2603.02398`](https://arxiv.org/abs/2603.02398), [`arXiv:2606.02480`](https://arxiv.org/abs/2606.02480)), Agarwal ([`arXiv:2609.13596`](https://arxiv.org/abs/2609.13596)), Medley–Gokul–Luu–Manolios ([`arXiv:2609.17533v2`](https://arxiv.org/abs/2609.17533)), Rudich–Rousseau ([`arXiv:2610.01639`](https://arxiv.org/abs/2610.01639)), Tomar ([`arXiv:2610.11823v3`](https://arxiv.org/abs/2610.11823)), Sidebottom ([`arXiv:2611.12907`](https://arxiv.org/abs/2611.12907)).

---

## 0. Главные прорывы 2025–2026, которые полностью изменили карту целей

Перед разбором таблиц — шесть критических фактов 2025–2026 годов, незнание которых гарантированно приведёт к атаке закрытых или мёртвых целей:

1. **$\langle 3,3,3 \rangle$ над $\mathbb{F}_2$ и $\mathbb{Z}$ ОКОНЧАТЕЛЬНО ЗАКРЫТ (октябрь 2026, `VERIFIED`):** Tejasvi Singh Tomar ([`arXiv:2610.11823v3`](https://arxiv.org/abs/2610.11823), 2026-10-19), опираясь на формализованную нижнюю оценку 22 над $\mathbb{Z}$ от Rudich–Rousseau ([`arXiv:2610.01639`](https://arxiv.org/abs/2610.01639), 2026-10-01) и оценки 21 над $\mathbb{F}_2$ от Wang ([`arXiv:2609.06725`](https://arxiv.org/abs/2609.06725)) / Yang et al. ([`arXiv:2609.18722`](https://arxiv.org/abs/2609.18722)), **машинно доказал в Lean 4**, что $R_{\mathbb{F}_2}(\langle 3,3,3 \rangle) = R_{\mathbb{Z}}(\langle 3,3,3 \rangle) = 23$. Атаковать ранг 22 для $3 \times 3$ над $\mathbb{F}_2$ или $\mathbb{Z}$ бессмысленно (открытым остаётся только ранг над $\mathbb{Q}/\mathbb{C}$: 22 против 23, и аддитивная сложность, доведённая до **55 сложений** в плоской модели и **51 сложения** в альтернативном базисе — Sidebottom [`arXiv:2611.12907`](https://arxiv.org/abs/2611.12907), 2026-10-17).
2. **$\langle 4,4,4 : 48 \rangle$ переведена из $\mathbb{C}$ в $\mathbb{Z}[1/2]$ (2025–2026, `VERIFIED`):** после прорыва AlphaEvolve ([`arXiv:2506.13131`](https://arxiv.org/abs/2506.13131), май 2025, ранг 48 над $\mathbb{C}$), Dumas–Pernet–Sedoglavic ([`arXiv:2506.13242v7`](https://arxiv.org/abs/2506.13242), обновлено 2026-07-27) нашли изотропию, проецирующую $\langle 4,4,4 : 48 \rangle$ и $\langle 3,4,7 : 63 \rangle$ в **$\mathbb{Z}[1/2]$** (действительно над любым кольцом с обратимой двойкой). В октябре 2026 G. Sidebottom ([`gsidebottom/mat_mul`](https://github.com/gsidebottom/mat_mul)) довёл число не-умножений до **201** (197 сложений + 4 деления на 2, проверено в Lean 4).
3. **Локальная жёсткость $\langle 4,4,4 \rangle$ (сентябрь 2026, `VERIFIED`):** N. F. Zaru ([`big-brain-zaru/FastMatrixF2`](https://github.com/big-brain-zaru/FastMatrixF2), DOI `10.5281/zenodo.22823114`, 2026-09-18) нашёл **вторую неэквивалентную** схему ранга 47 над $\mathbb{F}_2$ для $4 \times 4$, но доказал, что **обе** схемы ранга 47 **флип-изолированы** (0 общих множителей по всем трём осям) и **обструированы уже на первом шаге Гензеля ($\mathbb{Z}/4\mathbb{Z}$)**. Одновременно A. Agarwal ([`arXiv:2609.13596`](https://arxiv.org/abs/2609.13596), 2026-09-11) доказал, что вокруг рациональной 48-членной схемы $D(2)$ над $\mathbb{C}$ локальный ранговый радиус $\ge 12$ (любое укорочение требует одновременной замены $\ge 13$ ранговых слагаемых).
4. **$\langle 5,5,5 : 93 \rangle$ и $\langle 6,6,6 : 153 \rangle$ закрыты в $\mathbb{Z}_T$ (2025, `VERIFIED`):** Moosbauer–Poole ([`arXiv:2502.04514`](https://arxiv.org/abs/2502.04514), февраль 2025) с помощью симметричного flip-graph получили 93 и 153 над $\mathbb{Z}$, а Перминов ([`arXiv:2511.20317`](https://arxiv.org/abs/2511.20317), ноябрь 2025) нашёл их в чистом тернарном виде $\mathbb{Z}_T = \{-1, 0, 1\}$. Разрыв $\mathbb{Z}_2/\mathbb{Z}$ по $5 \times 5$ (который был $94$ vs $97$ в 2024 г.) полностью закрыт.
5. **Волна Meta Flip Graph + Serendipitous Product (октябрь 2025 — июнь 2026, `VERIFIED`):** Kauers–Wood ([`arXiv:2510.19787`](https://arxiv.org/abs/2510.19787)) улучшили ~30 форматов, а Перминов ([`arXiv:2511.20317`](https://arxiv.org/abs/2511.20317), [`arXiv:2603.02398`](https://arxiv.org/abs/2603.02398), [`arXiv:2606.02480`](https://arxiv.org/abs/2606.02480)) улучшил ранги **207 форматов** до $16^3$ и перевёл **375 из 680** форматов в $\mathbb{Z}_T$ (включая $\langle 4,4,10 : 115 \rangle$ с $\omega \approx 2.80479 < \log_2 7$ и $\langle 5,5,12 : 204 \rangle$ с $\omega \approx 2.79715$).
6. **GPU Flip Graph над $\mathbb{F}_2$ пробил $\langle 7,7,7 : 245 \rangle$ (октябрь 2026, `VERIFIED`):** Medley, Gokul, Luu & Manolios ([`arXiv:2609.17533v2`](https://arxiv.org/abs/2609.17533), 2026-10-26) достигли **$>10^9$ шагов/с** на H200 через warp-level matching и generalized multi-summand flips, доказали сильную связность $\mathbb{F}_2$-флип-графа **только через `flip` + `plus`** (без `reduce` в аргументе связности) и опустили $R_{\mathbb{F}_2}(\langle 7,7,7 \rangle)$ с $248 \to \mathbf{245}$ — при том что в $\mathrm{char}\,0$ рекорд $7\times 7$ всё ещё составной ($249$ над $\mathbb{Q}$, $250$ над $\mathbb{Z}_T$)!

> **Критическая сверка баз (`VERIFIED`):** По всем 102 форматам диапазона ($2 \le m \le n \le p \le 8$ и прямоугольники $m \le 4, n \le 5, p \le 10$) между `sedoglavic/fmm_digest/fmm_sota.json` (`2026-11-15`) и `dronperminov/FastMatrixMultiplication/schemes/status.json` (`2026-11-16`) зафиксировано **0 расхождений** по $R_{\mathbb{Q}}$. Однако **ни Lille, ни Perminov `status.json`, ни `solven-eu/matmulcatalog` по отдельности не содержат полного среза $R_{\mathbb{Z}_2}$**: критические $\mathbb{Z}_2$-рекорды (`4x5x5:73`, `4x5x6:89`, `4x4x8:94`, `7x7x7:245`, `7x7x8:275`, `7x8x8:302`) восстановлены напрямую из первоисточников (`arXiv:2312.16960`, `mkauers/matrix-multiplication/456`, `arXiv:2511.20317` Table 3, `arXiv:2609.17533v2`).

---

## 1. Живая матрица мировых рекордов 2026 года ($\langle m, n, p \rangle$)

**Обозначения колец:**
- $R_{\mathbb{Z}_2}$ — наименьший известный билинейный ранг над $\mathbb{F}_2 = \mathbb{Z}/2\mathbb{Z}$. Так как любая схема над $\mathbb{Z}_T$ или $\mathbb{Z}$ редуцируется по модулю $2$, всегда выполнено $R_{\mathbb{Z}_2} \le \min(R_{\mathbb{Z}_T}, R_{\mathbb{Z}})$. *(Внимание: схема над $\mathbb{Q}$ со знаменателем, кратным 2, например $\pm 1/2$ или $\pm 1/8$, **не** редуцируется в $\mathbb{F}_2$!)*
- $R_{\mathbb{Z}_T}$ — ранг при коэффициентах строго в тернарном множестве $\{-1, 0, 1\}$ (действительна над **любым** кольцом без делений и масштабирований).
- $R_{\mathbb{Z}}$ — ранг при произвольных целых коэффициентах (действительна над любым кольцом).
- $R_{\mathbb{Q}/\mathbb{C}}$ — ранг над $\mathbb{Q}$ (и $\mathbb{R}, \mathbb{C}$). В 2026 году **все** рекорды для $2 \le m \le n \le p \le 10$ реализованы над $\mathbb{Q}$ (комплексные схемы AlphaEvolve для $\langle 4,4,4:48 \rangle$ и $\langle 3,4,7:63 \rangle$ переведены в $\mathbb{Z}[1/2] \subset \mathbb{Q}$ в `arXiv:2506.13242`).
- $\omega_{\mathbb{Q}} = \frac{3 \ln R_{\mathbb{Q}}}{\ln(mnp)}$ (порог Штрассена $\log_2 7 \approx 2.807355$, порог AlphaEvolve $\log_4 48 \approx 2.792481$).
- Все форматы упорядочены канонически $m \le n \le p$ (ранг инвариантен относительно $S_3$-перестановок осей).
- `LB` — доказанная нижняя оценка ранга (из `solven-eu/matmulcatalog/docs/lower-bounds.json` + Tomar `arXiv:2610.11823v3`).

### 1.1. Ключевые малые и средние форматы ($3 \le m \le n \le p \le 8$)

| Формат | `max_flat` | `LB` (доказана) | $R_{\mathbb{Z}_2}$ | $R_{\mathbb{Z}_T}$ | $R_{\mathbb{Z}}$ | $R_{\mathbb{Q}/\mathbb{C}}$ | $\omega_{\mathbb{Q}}$ | Статус и точные источники (на 2026-11-22) |
|---|---:|---|---:|---:|---:|---:|---:|---|
| $\langle 3,3,3 \rangle$ | 9 | $\mathbb{F}_2,\mathbb{Z}: 23$ (tight!), $\text{all}: 19$ | **23** | **23** | **23** | **23** | 2.854050 | `VERIFIED`: Laderman 1976; **нижняя оценка 23 над $\mathbb{F}_2,\mathbb{Z}$ доказана в Lean 4** — Tomar ([arXiv:2610.11823v3](https://arxiv.org/abs/2610.11823), 2026-10-19); 55 слож. flat / 51 alt-basis ([arXiv:2611.12907](https://arxiv.org/abs/2611.12907)) |
| $\langle 3,3,4 \rangle$ | 12 | $\mathbb{F}_2: 25$ (Wang 2026) | 29 | 29 | 29 | 29 | 2.818985 | `VERIFIED`: Smirnov 2013; Lille (`2026-11-15`), Perminov `ZT` (`2026-11-16`) |
| $\langle 3,3,5 \rangle$ | 15 | — | 36 | 36 | 36 | 36 | 2.824142 | `VERIFIED`: Smirnov 2013 / AlphaTensor 2022; Perminov `ZT` (`2026-11-16`) |
| $\langle 3,3,6 \rangle$ | 18 | — | 42 | 42 | 42 | **40** | **2.774300** | `VERIFIED`: $\mathbb{Q}$ ($\pm 1/8$) — Smirnov 2013 (Lille `2026-11-15`); $\mathbb{Z}_T,\mathbb{Z},\mathbb{Z}_2 = 42$ — K&M 2023 / Perminov `arXiv:2603.02398` Sec 7.4 |
| $\langle 3,3,7 \rangle$ | 21 | — | 49 | 49 | 49 | 49 | 2.818026 | `VERIFIED`: Lille `<3,3,1:9>+<3,3,6:40>`; `ZT=49` — Perminov (`2026-11-16`) |
| $\langle 3,3,8 \rangle$ | 24 | — | 56 | 56 | 56 | **55** | 2.811068 | `VERIFIED`: $\mathbb{Q}$ ($\pm 1/8$) — Lille `<3,3,6:40>+<3,3,2:15>`; `ZT=56` — Perminov (`2026-11-16`) |
| $\langle 3,4,4 \rangle$ | 16 | $\mathbb{F}_2: 29$ (Wang 2026) | 38 | 38 | 38 | 38 | 2.821359 | `VERIFIED`: Smirnov 2013; Lille (`2026-11-15`), Perminov `ZT` (`2026-11-16`) |
| $\langle 3,4,5 \rangle$ | 20 | — | 47 | 47 | 47 | 47 | 2.821928 | `VERIFIED`: AlphaTensor 2022 (`DOI:10.1038/s41586-022-05172-4`); Perminov `ZT` (`arXiv:2511.20317`) |
| $\langle 3,4,6 \rangle$ | 24 | — | 54 | 54 | 54 | 54 | **2.798196** | `VERIFIED`: AlphaEvolve May 2025 (`arXiv:2506.13131`) & Kauers–Wood Oct 2025 (`arXiv:2510.19787`); Perminov `ZT` (`2026-11-16`) |
| $\langle 3,4,7 \rangle$ | 28 | — | 64 | 64 | 64 | **63** | **2.805217** | `VERIFIED`: $\mathbb{Q}$ ($\pm 1/2$) — AlphaEvolve May 2025 $\to$ Dumas–Pernet–Sedoglavic (`arXiv:2506.13242`); `ZT=Z=Z2=64` — Kauers–Wood Oct 2025 / Perminov (`2026-11-16`) |
| $\langle 3,4,8 \rangle$ | 32 | — | 73 | 73 | 73 | 73 | 2.819982 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); Perminov `ZT` (`arXiv:2603.02398`) |
| $\langle 3,5,5 \rangle$ | 25 | — | 58 | 58 | 58 | 58 | 2.819514 | `VERIFIED`: Sedoglavic & Smirnov 2021 / AlphaTensor 2022; Perminov `ZT` (`2026-11-16`) |
| $\langle 3,5,6 \rangle$ | 30 | — | 68 | 68 | 68 | 68 | 2.813124 | `VERIFIED`: AlphaEvolve May 2025 (`arXiv:2506.13131`); Perminov `ZT` (`arXiv:2603.02398`) |
| $\langle 3,5,7 \rangle$ | 35 | — | 79 | 79 | 79 | 79 | 2.816600 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); Perminov `ZT` (`arXiv:2603.02398`) |
| $\langle 3,5,8 \rangle$ | 40 | — | 90 | 90 | 90 | 90 | 2.819729 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); Perminov `ZT` (`arXiv:2603.02398`) |
| $\langle 3,6,6 \rangle$ | 36 | — | 82 | 82 | 82 | **80** | 2.807713 | `VERIFIED`: $\mathbb{Q}$ ($\pm 1/8$) — Lille `<1,2,1:2> ⊗ <3,3,6:40>`; `ZT=82` — Perminov (`2026-11-16`) |
| $\langle 3,6,7 \rangle$ | 42 | — | 96 | 96 | 96 | **94** | 2.818257 | `VERIFIED`: $\mathbb{Q}$ ($\pm 1/8, \pm 1/2$) — Lille `<3,6,3:40>+<3,6,4:54>`; `ZT=96` — Perminov (`2026-11-16`) |
| $\langle 3,6,8 \rangle$ | 48 | — | 108 | 108 | 108 | 108 | 2.826342 | `VERIFIED`: Lille `<1,1,2:2> ⊗ <3,6,4:54>` / Kauers–Wood Oct 2025; `ZT=108` — Perminov (`2026-11-16`) |
| $\langle 3,7,7 \rangle$ | 49 | — | 111 | 111 | 111 | 111 | 2.831135 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=111` — Perminov (`arXiv:2603.02398`) |
| $\langle 3,7,8 \rangle$ | 56 | — | 128 | 128 | 128 | **126** | 2.831567 | `VERIFIED`: $\mathbb{Q}$ ($\pm 1/2$) — Lille `<1,1,2:2> ⊗ <3,7,4:63>`; `ZT=128` — Perminov (`2026-11-16`) |
| $\langle 3,8,8 \rangle$ | 64 | — | **145** | 146 | 146 | **145** | 2.839794 | `VERIFIED`: $\mathbb{Z}_2=145$ — Kauers–Wood (`mkauers/388`) & Perminov (`arXiv:2511.20317` T.3); $\mathbb{Q}=145$ ($\pm 1/8$) — Lille `<3,8,3:55>+<3,8,5:90>`; `ZT=146` — Perminov (`2026-11-16`) |
| $\langle 4,4,4 \rangle$ | 16 | $\mathbb{R}: 34$ (LM 2017) | **47** | 49 | 49 | **48** | **2.792481** | `VERIFIED`: $\mathbb{Z}_2=47$ — AlphaTensor 2022 + Zaru 2026 (2-я неэкв.); $\mathbb{Q}=48$ ($\mathbb{Z}[1/2]$) — AlphaEvolve May 2025 $\to$ DPS (`arXiv:2506.13242`); `ZT=49` — Strassen 1969 |
| $\langle 4,4,5 \rangle$ | 20 | — | **60** | 61 | 61 | 61 | 2.814365 | `VERIFIED`: $\mathbb{Z}_2=60$ — Kauers–Moosbauer 2023 (`arXiv:2212.01175`); `ZT=Z=Q=61` — AlphaEvolve May 2025 / Perminov `ZT` (`arXiv:2511.20317`) |
| $\langle 4,4,6 \rangle$ | 24 | — | 73 | 73 | 73 | 73 | 2.819982 | `VERIFIED`: Smirnov 2023 ($\mathbb{Q}$) / Kauers–Wood Oct 2025; `ZT=73` — Perminov (`arXiv:2511.20317`) |
| $\langle 4,4,7 \rangle$ | 28 | — | 85 | 85 | 85 | 85 | 2.824617 | `VERIFIED`: AlphaEvolve May 2025 (`arXiv:2506.13131`); `ZT=85` — Perminov (`2026-11-16`) |
| $\langle 4,4,8 \rangle$ | 32 | — | **94** | 96 | 96 | **95** | 2.815652 | `VERIFIED`: $\mathbb{Z}_2=94$ — Perminov (`arXiv:2511.20317` T.3; также $2\times\langle 4,4,4:47 \rangle$); $\mathbb{Q}=95$ — Perminov 2026 (`status.json`); `ZT=96` — AlphaEvolve 2025 / Perminov |
| $\langle 4,5,5 \rangle$ | 25 | — | **73** | 76 | 76 | 76 | 2.821220 | `VERIFIED`: $\mathbb{Z}_2=73$ — **Arai–Ichikawa–Hukushima (ISSAC 2024, `arXiv:2312.16960` T.10)**; `ZT=Z=Q=76` — AlphaTensor 2022 / Lille (`2026-11-15`) / Perminov (`2026-11-16`) |
| $\langle 4,5,6 \rangle$ | 30 | — | **89** | 90 | 90 | 90 | 2.819729 | `VERIFIED`: $\mathbb{Z}_2=89$ — **Kauers–Wood Oct 2025 (`mkauers/456/*.mod2.exp`) & Perminov (`arXiv:2511.20317` T.3)**; `ZT=Z=Q=90` — AlphaEvolve May 2025 / Kauers–Wood / Perminov |
| $\langle 4,5,7 \rangle$ | 35 | — | 104 | 104 | 104 | 104 | 2.819543 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=104` — Perminov (`arXiv:2511.20317`) |
| $\langle 4,5,8 \rangle$ | 40 | — | 118 | 118 | 118 | 118 | 2.820013 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`) & Perminov `ZT` (`arXiv:2511.20317`, Lille `2026-11-15`) |
| $\langle 4,6,6 \rangle$ | 36 | — | 105 | 105 | 105 | 105 | 2.809337 | `VERIFIED`: Lille `<2,2,2:7> ⊗ <2,3,3:15>` (`ZT=105`, `2026-11-15`) |
| $\langle 4,6,7 \rangle$ | 42 | — | 123 | 123 | 123 | 123 | 2.817458 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=123` — Perminov (`arXiv:2511.20317`) |
| $\langle 4,6,8 \rangle$ | 48 | — | 140 | 140 | 140 | 140 | 2.819770 | `VERIFIED`: Lille `<2,2,2:7> ⊗ <2,3,4:20>` (`ZT=140`, `2026-11-15`) |
| $\langle 4,7,7 \rangle$ | 49 | — | 144 | 144 | 144 | 144 | 2.824766 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=144` — Perminov (`2026-11-16`) |
| $\langle 4,7,8 \rangle$ | 56 | — | 161 | 161 | 161 | 161 | 2.816927 | `VERIFIED`: Perminov 2026 (`arXiv:2603.02398` / `arXiv:2606.02480`, `ZT=161`); Lille (`2026-11-15`) |
| $\langle 4,8,8 \rangle$ | 64 | — | 180 | 180 | 180 | 180 | 2.809445 | `VERIFIED`: Perminov 2026 (`ZT=180`, `status.json 2026-11-16`); Lille (`2026-11-15`) |
| $\langle 5,5,5 \rangle$ | 25 | $\text{all}: 53$ (Bläser 2003) | 93 | 93 | 93 | 93 | 2.816262 | `VERIFIED`: **Moosbauer–Poole Feb 2025 (`arXiv:2502.04514`)**; `ZT=93` — Perminov (`arXiv:2511.20317`); Lille (`2026-11-15`) |
| $\langle 5,5,6 \rangle$ | 30 | — | 110 | 110 | 110 | 110 | 2.814302 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=110` — Perminov (`arXiv:2511.20317`) |
| $\langle 5,5,7 \rangle$ | 35 | — | 127 | 127 | 127 | 127 | 2.813778 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=127` — Perminov (`arXiv:2511.20317`) |
| $\langle 5,5,8 \rangle$ | 40 | — | 144 | 144 | 144 | 144 | 2.813995 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=144` — Perminov (`arXiv:2511.20317`) |
| $\langle 5,6,6 \rangle$ | 36 | — | 130 | 130 | 130 | 130 | 2.812002 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=130` — Perminov (`arXiv:2511.20317`) |
| $\langle 5,6,7 \rangle$ | 42 | — | 150 | 150 | 150 | 150 | 2.811222 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=150` — Perminov (`arXiv:2511.20317`) |
| $\langle 5,6,8 \rangle$ | 48 | — | 170 | 170 | 170 | 170 | 2.811241 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=170` — Perminov (`arXiv:2511.20317`) |
| $\langle 5,7,7 \rangle$ | 49 | — | 176 | 176 | 176 | 176 | 2.819619 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=176` — Perminov (`arXiv:2511.20317`) |
| $\langle 5,7,8 \rangle$ | 56 | — | 204 | 204 | 204 | 204 | 2.831403 | `VERIFIED`: Perminov 2026 (`ZT=204`, снижено с 205, `docs/results.md 2026-11-16`); Lille (`2026-11-15`) |
| $\langle 5,8,8 \rangle$ | 64 | — | 230 | 230 | 230 | 230 | 2.828247 | `VERIFIED`: **чисто составная** в Lille (`3*<2,4,4:26> + 4*<3,4,4:38>`, `2026-11-15`); Perminov `ZT=230` (`2026-11-16`) |
| $\langle 6,6,6 \rangle$ | 36 | $\text{all}: 76$ (Bläser 2003) | 153 | 153 | 153 | 153 | 2.807541 | `VERIFIED`: **Moosbauer–Poole Feb 2025 (`arXiv:2502.04514`)**; `ZT=153` — Perminov (`arXiv:2511.20317`); Lille (`2026-11-15`) |
| $\langle 6,6,7 \rangle$ | 42 | — | 183 | 183 | 183 | 183 | 2.826414 | `VERIFIED`: Kauers–Wood Oct 2025 (`arXiv:2510.19787`); `ZT=183` — Perminov (`2026-11-16`); Lille (`2026-11-15`) |
| $\langle 6,6,8 \rangle$ | 48 | — | 203 | 203 | 203 | 203 | 2.814715 | `VERIFIED`: **чисто составная** в Lille (`<2,2,2:7> ⊗ <3,3,4:29>`, `2026-11-15`); Perminov `ZT=203` (`2026-11-16`) |
| $\langle 6,7,7 \rangle$ | 49 | — | 212 | 212 | 212 | 212 | 2.827401 | `VERIFIED`: Perminov 2026 (`ZT=212`, снижено с 215, `docs/results.md 2026-11-16`); Lille (`2026-11-15`) |
| $\langle 6,7,8 \rangle$ | 56 | — | 238 | 238 | 238 | 238 | 2.822159 | `VERIFIED`: Perminov 2026 (`ZT=238`, снижено с 239, `docs/results.md 2026-11-16`); Lille (`2026-11-15`) |
| $\langle 6,8,8 \rangle$ | 64 | — | 266 | 266 | 266 | 266 | 2.814904 | `VERIFIED`: **чисто составная** в Lille (`<2,2,2:7> ⊗ <3,4,4:38>`, `2026-11-15`); Perminov `ZT=266` (`2026-11-16`) |
| $\langle 7,7,7 \rangle$ | 49 | $\text{all}: 103$ (Bläser 2003) | **245** | 250 | 250 | **249** | 2.835410 | `VERIFIED`: **$\mathbb{Z}_2=245$ — Medley–Gokul–Luu–Manolios Oct 2026 (`arXiv:2609.17533v2`)**; $\mathbb{Q}=249$ составная (`<4,4,4:48>+3<3,3,4:29>+3<3,4,4:38>`); `ZT=250` составная |
| $\langle 7,7,8 \rangle$ | 56 | — | **275** | 278 | 278 | **277** | 2.825542 | `VERIFIED`: $\mathbb{Z}_2=275$ — Perminov (`arXiv:2511.20317` T.3); $\mathbb{Q}=277$ составная (`<3,3,4:29>+2<4,4,4:48>+4<3,4,4:38>`); `ZT=278` |
| $\langle 7,8,8 \rangle$ | 64 | — | **302** | 310 | 310 | **306** | 2.812668 | `VERIFIED`: $\mathbb{Z}_2=302$ — Perminov (`arXiv:2511.20317` T.3); $\mathbb{Q}=306$ составная (`3<3,4,4:38>+4<4,4,4:48>`); `ZT=310` |
| $\langle 8,8,8 \rangle$ | 64 | $\text{all}: 136$ (Bläser/Landsberg) | **329** | 343 | 343 | **336** | **2.797439** | `VERIFIED`: $\mathbb{Z}_2=329$ ($7 \times 47$, Perminov `arXiv:2511.20317` T.3); $\mathbb{Q}=336$ ($7 \times 48$, Lille `<2,2,2:7> ⊗ <4,4,4:48>`); `ZT=343` ($7^3$) |

### 1.2. Базовые форматы $m = 2$ ($2 \le n \le p \le 8$) и перспективные прямоугольники ($m \le 4, p \in \{9, 10\}$)

| Формат | `max_flat` | `LB` | $R_{\mathbb{Z}_2}$ | $R_{\mathbb{Z}_T}$ | $R_{\mathbb{Z}}$ | $R_{\mathbb{Q}/\mathbb{C}}$ | $\omega_{\mathbb{Q}}$ | Статус и источник (на 2026-11-22) |
|---|---:|---|---:|---:|---:|---:|---:|---|
| $\langle 2,2,2 \rangle$ | 4 | $\text{all}: 7$ (tight) | 7 | 7 | 7 | 7 | 2.807355 | `VERIFIED`: Strassen 1969; оптимальность Winograd / Hopcroft–Kerr 1971 |
| $\langle 2,2,3..8 \rangle$ | 6..16 | $2,2,3..5$ tight | 11,14,18,21,25,28 | 11,14,18,21,25,28 | 11..28 | 11..28 | 2.855–2.898 | `VERIFIED`: Hopcroft–Kerr 1971 ($R = \lceil 7p/2 \rceil$); Lille / Perminov `ZT` (`2026-11-16`) |
| $\langle 2,3,3..8 \rangle$ | 9..24 | $2,3,3:15$; $2,3,4:\text{char0 }20$ | 15,20,25,30,35,40 | 15..40 | 15..40 | 15..40 | 2.811–2.859 | `VERIFIED`: Hopcroft–Kerr 1971 ($R = 5p$); Lille / Perminov `ZT` (`2026-11-16`) |
| $\langle 2,4,4 \rangle$ | 16 | — | 26 | 26 | 26 | 26 | 2.820264 | `VERIFIED`: Hopcroft–Kerr 1971; Perminov `ZT` (`2026-11-16`) |
| $\langle 2,4,5 \rangle$ | 20 | — | 33 | 33 | 33 | **32** | 2.818527 | `VERIFIED`: $\mathbb{Q}=32$ ($\pm 1/2$) — AlphaEvolve May 2025; `ZT=Z=Z2=33` — Hopcroft–Kerr 1971 / Perminov (`arXiv:2603.02398` Sec 7.4) |
| $\langle 2,4,6..8 \rangle$ | 24..32 | — | 39, 45, 51 | 39, 45, 51 | 39, 45, 51 | 39, 45, 51 | 2.836–2.839 | `VERIFIED`: HK 1971 (`39`), AlphaEvolve May 2025 (`45, 51`); все в `ZT` — Perminov (`2026-11-16`) |
| $\langle 2,5,5..8 \rangle$ | 25..40 | — | 40, 47, 55, 63 | 40, 47, 55, 63 | 40..63 | 40..63 | 2.821–2.836 | `VERIFIED`: HK 1971 (`40`), AlphaEvolve 2025 (`47`), Kauers–Wood Oct 2025 (`55, 63`); все в `ZT` — Perminov (`2026-11-16`) |
| $\langle 2,6,6..8 \rangle$ | 36..48 | — | 56, 66, 75 | 56, 66, 75 | 56, 66, 75 | 56, 66, 75 | 2.824–2.838 | `VERIFIED`: K&M 2023 (`56`), Kauers–Wood Oct 2025 (`66, 75`); все в `ZT` — Perminov (`2026-11-16`) |
| $\langle 2,7,7..8 \rangle, \langle 2,8,8 \rangle$ | 49..64 | — | 76, 88, 100 | 76, 88, 100 | 76, 88, 100 | 76, 88, 100 | 2.834–2.847 | `VERIFIED`: Kauers–Wood Oct 2025 (`76`), HK 1971 (`88, 100`); все в `ZT` — Perminov (`2026-11-16`) |
| $\langle 2,4,9 \rangle, \langle 2,4,10 \rangle$ | 36, 40 | — | 58, 64 | 58, 64 | 58, 64 | 58, 64 | 2.848, 2.847 | `VERIFIED`: Lille `<2,4,4:26>+<2,4,5:32>` / $2\times\langle 2,4,5:32\rangle$; закрыты в `ZT` — Perminov 2026 (`status.json`) |
| $\langle 3,3,9 \rangle, \langle 3,3,10 \rangle$ | 27, 30 | — | 63, 69 | 63, 69 | 63, 69 | 63, 69 | 2.828, 2.823 | `VERIFIED`: Lille (`<3,3,3:23>+<3,3,6:40>`, `<3,3,4:29>+<3,3,6:40>`); закрыты в `ZT` — Perminov 2026 |
| $\langle 3,4,9 \rangle, \langle 3,4,10 \rangle$ | 36, 40 | — | 83, 92 | 83, 92 | 83, 92 | 83, 92 | 2.831, 2.834 | `VERIFIED`: Lille (`<3,4,3:29>+<3,4,6:54>`, `<3,4,3:29>+<3,4,7:63>`); закрыты в `ZT` — Perminov 2026 |
| $\langle 3,5,9 \rangle, \langle 3,5,10 \rangle$ | 45, 50 | — | 102, 114 | 102, 114 | 102, 114 | 102, 114 | 2.829, 2.836 | `VERIFIED`: Perminov 2026 (`ZT=102` снижено со 104, `ZT=114` со 115, `docs/results.md 2026-11-16`) |
| $\langle 4,4,9 \rangle$ | 36 | — | 107 | 107 | 107 | **104** | **2.803561** | `VERIFIED`: $\mathbb{Q}=104$ (Smirnov 2022, знаменатели `2,3,8,9,16`); `ZT=Z=Z2=107` — Perminov (`2026-11-16`) |
| $\langle 4,4,10 \rangle$ | 40 | — | 115 | 115 | 115 | 115 | **2.804790** | `VERIFIED`: **Perminov Mar 2026 (`arXiv:2603.02398`, `ZT=115`, ниже экспоненты Штрассена!)**; Lille (`2026-11-15`) |
| $\langle 4,5,9 \rangle, \langle 4,5,10 \rangle$ | 45, 50 | — | 132, 146 | 132, 146 | 132, 146 | 132, 146 | 2.821, 2.822 | `VERIFIED`: Perminov Jun 2026 (`arXiv:2606.02480`, `ZT=132, 146` — побит даже ноябрьский $\mathbb{Z}_2=133$ для `4x5x9`!) |

---

## 2. Track A — Разрыв подъёма $\mathbb{Z}_2$ vs $\mathbb{Z}_T / \mathbb{Z} / \mathbb{Q}$ (МАКСИМАЛЬНЫЙ РЫЧАГ)

### 2.1. Полный реестр ОТКРЫТЫХ разрывов $R_{\mathbb{Z}_2} < R_{\mathbb{Z}_T / \mathbb{Z} / \mathbb{Q}}$ (на 2026-11-22)

В диапазоне $2 \le m \le n \le p \le 8$ (+ прямоугольники $m \le 4, n \le 5, p \le 10$) после волны подъёмов Перминова 2025–2026 осталось ровно **10 открытых форматов** с подтверждённым строгим неравенством $R_{\mathbb{Z}_2} < R_{\mathbb{Z}_T}$ (из них **9** со строгим неравенством $R_{\mathbb{Z}_2} < R_{\mathbb{Q}}$):

| # | Формат | `max_flat` | $R_{\mathbb{Z}_2}$ (источник) | $R_{\mathbb{Q}}$ | $R_{\mathbb{Z}_T}=R_{\mathbb{Z}}$ | Разрыв к $\mathbb{Q}$ | Разрыв к $\mathbb{Z}_T$ | Диагноз обструкции и рычаг |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 1 | **$\langle 4,5,5 \rangle$** | **25** | **73** (Arai et al., ISSAC 2024, `arXiv:2312.16960`) | 76 | 76 | **3** | **3** | **ГЛАВНАЯ ЦЕЛЬ TRACK A.** Единственный малый формат с разрывом $\ge 3$. Известные $\mathbb{Z}_2$-схемы 73–75 из [Arai 2024] не поднялись наивным Гензелем, но пространство орбит ранга 74–75 почти не прочёсывалось с mod-4-оракулом! |
| 2 | **$\langle 4,5,6 \rangle$** | **30** | **89** (Kauers–Wood Oct 2025 `mkauers/456`; Perminov Nov 2025 `arXiv:2511.20317`) | 90 | 90 | **1** | **1** | **ВЫСОКИЙ ШАНС.** В `mkauers/456` сохранены лишь 2 `.mod2.exp`-схемы. Крошечный размер (`uint32`) позволяет перебрать тысячи неизоморфных орбит ранга 89 с фильтром mod-4. |
| 3 | **$\langle 4,4,5 \rangle$** | **20** | **60** (Kauers–Moosbauer 2023, `arXiv:2212.01175`) | 61 | 61 | **1** | **1** | **ПРЯМОЙ ВЫЗОВ ALPHAEVOLVE.** Подъём до 60 в $\mathbb{Q}$ или $\mathbb{Z}_T$ опускает $\omega$ до **2.803049** ($< \log_2 7$!). Нужно пробовать не только $\mathbb{Z}_2$, но и $\mathbb{Z}_3/\mathbb{Z}_5$ (ради $\mathbb{Z}[1/2]$). |
| 4 | **$\langle 4,4,8 \rangle$** | **32** | **94** (Perminov Nov 2025, `arXiv:2511.20317` T.3) | 95 | 96 | **1** | **2** | **ДВОЙНАЯ СТУПЕНЬ.** $\mathbb{Q}=95$ имеет знаменатели `2,3,4,5`. Даже перевод 95 в $\mathbb{Z}_T$ — рекорд кольца, а подъём 94 — абсолютный мировой рекорд ($\omega=2.8091$). |
| 5 | **$\langle 7,7,7 \rangle$** | **49** | **245** (Medley et al., Oct 2026, `arXiv:2609.17533v2`) | 249 | 250 | **4** | **5** | **КРУПНЫЙ ГИБРИД TRACK A+B.** В $\mathrm{char}\,0$ рекорд до сих пор составной ($48+3\cdot 29+3\cdot 38$). Любой подъём на уровне 245–248 сносит рекорд Lille (`249`)! |
| 6 | **$\langle 7,7,8 \rangle$** | 56 | **275** (Perminov Nov 2025, `arXiv:2511.20317` T.3) | 277 | 278 | **2** | **3** | Составная в $\mathrm{char}\,0$ (`29+2·48+4·38`), гребень $+5.0$. Высокий потенциал как для подъёма, так и для прямой $\mathbb{Z}_T$-оптимизации. |
| 7 | $\langle 7,8,8 \rangle$ | 64 | **302** (Perminov Nov 2025, `arXiv:2511.20317` T.3) | 306 | 310 | **4** | **8** | Составная в $\mathrm{char}\,0$ (`3·38+4·48`), граница `uint64`. |
| 8 | $\langle 8,8,8 \rangle$ | 64 | **329** ($7 \times 47$, Perminov Nov 2025) | 336 | 343 | **7** | **14** | Кронекерова производная от $\langle 4,4,4 \rangle$ ($7 \times 47$ vs $7 \times 48$ vs $7 \times 49$); улучшение зависит от атомарного прорыва на $8\times 8$ или $4\times 4$. |
| 9 | $\langle 3,8,8 \rangle$ | 64 | **145** (Kauers–Wood `mkauers/388`; Perminov `arXiv:2511.20317`) | 145 | 146 | 0 | **1** | $\mathbb{Z}_2 = \mathbb{Q} = 145$, но $\mathbb{Q}$-схема составная со знаменателем $1/8$, а $\mathbb{Z}_T$ застрял на 146. |
| 10 | $\langle 4,4,4 \rangle$ | 16 | **47** (AlphaTensor 2022 + Zaru Sep 2026) | 48 | 49 | **1** | **2** | **MOONSHOT.** Обе известные $\mathbb{F}_2$-схемы ранга 47 флип-изолированы и обструированы $\bmod\,4$ (Zaru 2026); $D(2)$ ранга 48 имеет радиус $\ge 12$ (Agarwal 2026). |

### 2.2. Полный реестр ОТКРЫТЫХ рациональных разрывов $R_{\mathbb{Q}} < R_{\mathbb{Z}_T} = R_{\mathbb{Z}}$ (эффект знаменателей $1/2^k$)

Перекрёстный аудит 102 форматов выявил ровно **15 форматов**, где схема над $\mathbb{Q}$ строго лучше $\mathbb{Z}_T$ и $\mathbb{Z}$ (причём **ни в одном** из 102 форматов сейчас нет разрыва $R_{\mathbb{Z}} < R_{\mathbb{Z}_T}$ — Перминов полностью уравнял $\mathbb{Z}_T$ и $\mathbb{Z}$ до $p \le 10$!):

| Формат | $R_{\mathbb{Q}}$ | Знаменатели в $\mathbb{Q}$-схеме (`VERIFIED`, `status.json`) | $R_{\mathbb{F}_3}$ | $R_{\mathbb{Z}_2}$ | $R_{\mathbb{Z}_T}=R_{\mathbb{Z}}$ | Разрыв $\mathbb{Z}_T - \mathbb{Q}$ | Природа схемы |
|---|---:|---|---:|---:|---:|---:|---|
| $\langle 2,4,5 \rangle$ | **32** | $\pm 1/2$ ($\mathbb{Z}[1/2]$) | 32 | 33 | 33 | **1** | Атомарная (AlphaEvolve 2025); отмечена Перминовым (`arXiv:2603.02398` Sec 7.4) |
| $\langle 3,3,6 \rangle$ | **40** | $\pm 1/8$ ($\mathbb{Z}[1/2]$) | 40 | 42 | 42 | **2** | Атомарная (Smirnov 2013, $\omega=2.7743$); главная загадка `Sec 7.4` |
| $\langle 3,3,8 \rangle$ | **55** | $\pm 1/8$ | 55 | 56 | 56 | **1** | Составная `<3,3,6:40>+<3,3,2:15>` |
| $\langle 3,4,7 \rangle$ | **63** | $\pm 1/2$ ($\mathbb{Z}[1/2]$) | 63 | 64 | 64 | **1** | Атомарная (AlphaEvolve $\to$ DPS `arXiv:2506.13242`) |
| $\langle 3,6,6 \rangle$ | **80** | $\pm 1/8$ | 80 | 82 | 82 | **2** | Составная $2 \times \langle 3,3,6:40 \rangle$ |
| $\langle 3,6,7 \rangle$ | **94** | $\pm 1/8, \pm 1/2$ | 94 | 96 | 96 | **2** | Составная `<3,6,3:40>+<3,6,4:54>` |
| $\langle 3,7,8 \rangle$ | **126** | $\pm 1/2$ | 126 | 128 | 128 | **2** | Составная $2 \times \langle 3,7,4:63 \rangle$ |
| $\langle 3,8,8 \rangle$ | **145** | $\pm 1/8$ | 145 | 145 | 146 | **1** | Составная `<3,8,3:55>+<3,8,5:90>` (в $\mathbb{Z}_2$ есть 145!) |
| $\langle 4,4,4 \rangle$ | **48** | $\pm 1/2, \pm 1/4, \pm 1/8$ ($\mathbb{Z}[1/2]$) | 48 | 47 | 49 | **1** | Атомарная (AlphaEvolve $\to$ DPS `arXiv:2506.13242`) |
| $\langle 4,4,8 \rangle$ | **95** | $2, 3, 4, 5$ | 96 | 94 | 96 | **1** | Атомарная (Perminov 2026) |
| $\langle 4,4,9 \rangle$ | **104** | $2, 3, 8, 9, 16$ | 107 | 107 | 107 | **3** | Атомарная (Smirnov 2022, $\omega=2.80356$); рядом `4x4x10:115` в `ZT`! |
| $\langle 7,7,7 \rangle$ | **249** | $\pm 1/2 \dots \pm 1/8$ | 249 | 245 | 250 | **1** | Составная через $\langle 4,4,4:48 \rangle$ |
| $\langle 7,7,8 \rangle$ | **277** | $\pm 1/2 \dots \pm 1/8$ | 277 | 275 | 278 | **1** | Составная через $2 \times \langle 4,4,4:48 \rangle$ |
| $\langle 7,8,8 \rangle$ | **306** | $\pm 1/2 \dots \pm 1/8$ | 306 | 302 | 310 | **4** | Составная через $4 \times \langle 4,4,4:48 \rangle$ |
| $\langle 8,8,8 \rangle$ | **336** | $\pm 1/2 \dots \pm 1/8$ | 336 | 329 | 343 | **7** | Кронекерова $7 \times \langle 4,4,4:48 \rangle$ |

**Алгебраическая причина (почему $\mathbb{Z}_2$-поиск слеп к $\mathbb{Z}[1/2]$):** как показали Kauers–Wood ([`arXiv:2510.19787`](https://arxiv.org/abs/2510.19787)) и Perminov ([`arXiv:2603.02398`](https://arxiv.org/abs/2603.02398) Sec 7.4–7.5), любой рациональный алгоритм со знаменателем, делящимся на 2 (как $\langle 4,4,4:48 \rangle$, $\langle 3,3,6:40 \rangle$, $\langle 2,4,5:32 \rangle$, $\langle 3,4,7:63 \rangle$), **не имеет гомоморфного образа в $\mathbb{F}_2$**, поэтому **принципиально недостижим** гензелевым подъёмом из $\mathbb{Z}_2$. Зато $2^{-1} \equiv 2 \pmod{3}$ и $2^{-1} \equiv 3 \pmod{5}$ существуют в $\mathbb{F}_3$ и $\mathbb{F}_5$, что объясняет колонку $R_{\mathbb{F}_3} = R_{\mathbb{Q}}$ в таблице 2.2 и мотивирует добавление **$\mathbb{F}_3$-флип-графа + 3-адического Гензеля ($\mathbb{Z}_3 \to \mathbb{Z}_9 \to \mathbb{Z}_{27} \dots$)** в наш движок!

### 2.3. Разрывы, УЖЕ ЗАКРЫТЫЕ в 2025–2026 (НЕ атаковать повторно!)

- **$\langle 5,5,5 \rangle$:** был разрыв $94$ ($\mathbb{Z}_2$, Arai 2024) vs $97$ ($\mathbb{Z}$, K&M 2023) $\to$ **ЗАКРЫТ И ПОБИТ**: $93$ в $\mathbb{Z}_T$ (Moosbauer–Poole Feb 2025 + Perminov Nov 2025).
- **$\langle 6,6,6 \rangle$:** был $164$ ($\mathbb{Z}_2$, K&M 2023) $\to$ **ЗАКРЫТ И ПОБИТ**: $153$ в $\mathbb{Z}_T$ (Moosbauer–Poole Feb 2025 + Perminov Nov 2025).
- **$\langle 4,5,9 \rangle$ и $\langle 4,5,10 \rangle$:** в ноябре 2025 (`arXiv:2511.20317` T.3) Перминов имел разрыв $\mathbb{Z}_2=133 < \mathbb{Z}=136$ и $\mathbb{Z}_2=146 < \mathbb{Z}=151$ $\to$ **ЗАКРЫТЫ** в `arXiv:2606.02480`: сейчас $R_{\mathbb{Z}_T} = 132$ и $R_{\mathbb{Z}_T} = 146$!
- **$\langle 4,4,6 \rangle, \langle 2,4,9..10 \rangle, \langle 3,4,6 \rangle, \langle 3,4,8 \rangle, \langle 3,5,7..8 \rangle, \langle 4,5,7..8 \rangle, \langle 4,6,7 \rangle, \langle 4,7,7 \rangle, \langle 5,5,6..8 \rangle, \langle 5,6,6..8 \rangle, \langle 5,7,7 \rangle, \langle 6,6,7 \rangle$:** все прежние $\mathbb{Q}$-схемы Smirnov / AlphaEvolve / Kauers–Wood **уже переоткрыты в чистом $\mathbb{Z}_T$** Перминовым в 2025–2026 (`status.json 2026-11-16`).

---

## 3. Track B — Открытые цели прямого снижения ранга ($m, n, p \in [4..8]$)

Чтобы не искать вслепую в уже насыщенных точках (таких как $\langle 6,6,6:153 \rangle$ с $\omega = 2.80754$, всего $+0.05$ выше скейлинга Штрассена), мы просканировали все 35 форматов куба $[4..8]^3$ по четырём объективным индикаторам неэффективности (расчёт в Python по базе `2026-11-16`):
1. **Избыток над скейлингом Штрассена:** $\Delta_S = R_{\mathbb{Q}} - (mnp)^{\log_2(7)/3}$;
2. **Аномалия выпуклого гребня (convexity ridge):** $\Delta_{\text{ridge}} = R(m,n,p) - \max_{\text{ось } a} \frac{1}{2}\bigl(R(a{-}1,\dots) + R(a{+}1,\dots)\bigr)$ — показывает «горб» относительно непосредственных соседей;
3. **Составной статус в Lille:** не было ни одной атомарной схемы, побившей блочную/кронекерову сумму;
4. **Попадание в один 64-битный регистр (`uint64`):** $\max(mn, np, mp) \le 64$ — критично для warp-level GPU-ядра на RTX 5060.

### Топ аномалий Track B в кубе $[4..8]^3$ (все $\max\_\text{flat} \le 64$)

| Формат | `max_flat` | $R_{\mathbb{Z}_2}$ | $R_{\mathbb{Z}_T}$ | $R_{\mathbb{Q}}$ | $\omega_{\mathbb{Q}}$ | $\Delta_S$ | $\Delta_{\text{ridge}}$ (соседи $\to$ середина) | Природа текущего рекорда |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **$\langle 5,8,8 \rangle$** | **64** | 230 | 230 | 230 | 2.828247 | $+9.06$ | **`+7.0`** (`4x8x8:180`, `6x8x8:266` $\to 223.0$) | **Чисто составная** `3<2,4,4:26>+4<3,4,4:38>`! |
| **$\langle 6,6,7 \rangle$** | **42** | 183 | 183 | 183 | 2.826414 | $+6.32$ | **`+5.0`** (`6x6x6:153`, `6x6x8:203` $\to 178.0$) | Kauers–Wood Oct 2025; резкий скачок $+30$ от $6\times 6\times 6$ |
| **$\langle 7,7,8 \rangle$** | **56** | 275 | 278 | 277 | 2.825542 | $+9.85$ | **`+5.0`** (`6x7x8:238`, `8x7x8:306` $\to 272.0$) | **Чисто составная** в $\mathrm{char}\,0$; разрыв с $\mathbb{Z}_2=275$ |
| **$\langle 7,8,8 \rangle$** | **64** | 302 | 310 | 306 | 2.812668 | $+3.29$ | **`+5.0`** (`6x8x8:266`, `8x8x8:336` $\to 301.0$) | **Чисто составная** в $\mathrm{char}\,0$; разрыв с $\mathbb{Z}_2=302$ |
| **$\langle 7,7,7 \rangle$** | **49** | **245** | 250 | 249 | **2.835410** | **`+13.23`** | **`+4.5`** (`6x7x7:212`, `7x7x8:277` $\to 244.5$) | **Худшая $\omega$ в $[4..8]^3$**; составная в $\mathrm{char}\,0$, $\mathbb{Z}_2=245$! |
| **$\langle 5,7,8 \rangle$** | **56** | 204 | 204 | 204 | **2.831403** | $+9.01$ | **`+4.5`** (`4x7x8:161`, `6x7x8:238` $\to 199.5$) | **2-я худшая $\omega$**; активное падение `205->204` в 2026 |
| $\langle 6,7,8 \rangle$ | 56 | 238 | 238 | 238 | 2.822159 | $+6.73$ | $+3.5$ (`6x6x8:203`, `6x8x8:266` $\to 234.5$) | Активное падение `239->238` в 2026 (Perminov) |
| $\langle 4,7,7 \rangle$ | 49 | 144 | 144 | 144 | 2.824766 | $+4.34$ | $+2.0$ (`4x6x7:123`, `4x7x8:161` $\to 142.0$) | Kauers–Wood Oct 2025 |
| $\langle 6,7,7 \rangle$ | 49 | 212 | 212 | 212 | 2.827401 | $+7.90$ | $+1.5$ (`6x6x7:183`, `6x7x8:238` $\to 210.5$) | Активное падение `215->212` в 2026 (Perminov) |
| $\langle 4,4,7 \rangle$ | 28 | 85 | 85 | 85 | 2.824617 | $+2.28$ | $+1.5$ (`3x4x7:63`, `4x5x7:104` $\to 83.5$) | AlphaEvolve May 2025 (маленький `max_flat=28`) |

**Ключевой вывод Track B:** Форматы **$\langle 6,6,7 \rangle$**, **$\langle 5,8,8 \rangle$**, **$\langle 7,7,7 \rangle$** и **$\langle 5,7,8 \rangle$** обладают одновременным сочетанием огромного гребня ($\Delta_{\text{ridge}} \in [+4.5, +7.0]$), высокой $\omega \ge 2.826$, нетронутой составной структуры или свежего 2026-градиента снижения, и при этом полностью укладываются в одно 64-битное машинное слово! Именно здесь комбинация **двунаправленного мета-флипа** (проекция сверху + расширение снизу), **generalized multi-summand flips** (`arXiv:2609.17533`) и **`SLEEP-DAG-RSI` (`plus`-туннелирование + структурный `CROSSOVER`)** даёт максимальную вероятность немедленного снижения на `-1..-4` умножения.

---

## 4. Track C — Верификация, нотация и стандарты публикации в 2026 году

### 4.1. Критическая ловушка транспонирования ($C = AB$ против $\mathrm{tr}(ABC)$)

> [!CAUTION]
> **Обязательное предупреждение для `brent_verifier.py` (`@fmm-architect`)!**
> На квадратных матрицах $m = n = p$ ошибка транспонирования $C \leftrightarrow C^\top$ легко пропустить при неполных тестах; на прямоугольных $m \ne p$ (например $\langle 2,3,4 \rangle$) несовпадение конвенций немедленно ломает размерности или выдаёт неверный тензор!

В литературе и репозиториях 2026 года одновременно живут **две** конвенции записи билинейной схемы $\langle m, n, p : R \rangle$ для произведения $C = A \cdot B$ ($A \in K^{m \times n}$, $B \in K^{n \times p}$, $C \in K^{m \times p}$):

1. **Прямая (вычислительная) конвенция $C = AB$** (используется в `khoruzhii/lita`, `arXiv:2609.17533`, `PLinOpt` в форме $P(LA \odot RB)$ и в [Perminov `arXiv:2511.20317`, ур. (1)]):
   $$m_r = \left(\sum_{i_1=1}^{m}\sum_{i_2=1}^{n} U_{r,(i_1,i_2)} A_{i_1,i_2}\right)\left(\sum_{j_1=1}^{n}\sum_{j_2=1}^{p} V_{r,(j_1,j_2)} B_{j_1,j_2}\right), \qquad C_{k_1,k_2} = \sum_{r=1}^{R} W_{r,(k_1,k_2)} m_r.$$
   Тройка факторов имеет формы $U \in K^{R \times mn}$, $V \in K^{R \times np}$, $W \in K^{R \times mp}$, а **уравнения Брента** имеют вид:
   $$\sum_{r=1}^{R} U_{r,(i_1,i_2)}\, V_{r,(j_1,j_2)}\, W_{r,(k_1,k_2)} = \delta_{i_2, j_1}\, \delta_{i_1, k_1}\, \delta_{j_2, k_2}, \quad \forall i_1,k_1 \in [m],\; i_2,j_1 \in [n],\; j_2,k_2 \in [p].$$

2. **Следовая (циклически-симметричная) конвенция $\mathrm{tr}(A B C_{\text{dual}})$ / $C^\top$** (используется в **Université de Lille** `.tensor.mpl`, **Perminov `scheme.json`**, **Kauers–Moosbauer `.exp` / `.m`** — `VERIFIED` напрямую по `https://fmm.univ-lille.fr/2x3x4_tensor.mpl.bz2` и `dronperminov/FastMatrixMultiplication/README.md`!):
   $$\mathcal{M}_{m,n,p} = \sum_{i=1}^{m}\sum_{j=1}^{n}\sum_{k=1}^{p} E^{(m\times n)}_{i,j} \otimes E^{(n\times p)}_{j,k} \otimes E^{(p\times m)}_{k,i} = \sum_{r=1}^{R} \tilde{U}^{(r)} \otimes \tilde{V}^{(r)} \otimes \tilde{W}^{(r)}.$$
   Здесь $\tilde{W}^{(r)} \in K^{p \times m}$ кодирует **транспонированный выход $C^\top$** (строки индексированы по $k_2 \in [p]$, столбцы по $k_1 \in [m]$!), так что:
   $$C_{k_1,k_2} = \sum_{r=1}^{R} \tilde{W}^{(r)}_{k_2, k_1} m_r \quad \Longleftrightarrow \quad \sum_{r=1}^{R} U_{r,(i_1,i_2)}\, V_{r,(j_1,j_2)}\, \tilde{W}_{r,(k_2,k_1)} = \delta_{i_2, j_1}\, \delta_{j_2, k_2}\, \delta_{k_1, i_1}.$$
   - В Lille `2x3x4_tensor.mpl.bz2` (`VERIFIED`): $A$ имеет размер $2\times 3$, $B$ — $3\times 4$, а третий фактор `C` в `Triad` имеет размер **$4 \times 2$**, и итоговая проверка в Maple явно применяет `LinearAlgebra:-Transpose(op([1,i,1,3],Tensor))`!
   - В Perminov `README.md` (`VERIFIED`): поле `w` имеет размер `m × (n₃·n₁)` и «encodes the linear form of **`Cᵀ`**».
   - В Perminov `LRP` vs `PLinOpt` (`VERIFIED` по предупреждению в `2x3x4_LRP.mpl.bz2`): файл Lille `_LRP.mpl.bz2` явно содержит пометку `Warning: the following LRP representation does not follow the plinopt encoding convention`.

**Инженерное правило для нашего конвейера:** внутри движка Flip-Graph и в `brent_verifier.py` фиксировать каноническую конвенцию явно флагом (`convention="direct"` vs `convention="trace"`), а при любом импорте/экспорте из Lille / Perminov / Kauers–Moosbauer выполнять детерминированную транспозицию блока $W$ ($p \times m \leftrightarrow m \times p$) и прогонять сквозной тест на прямоугольной матрице $\langle 2,3,4 \rangle$!

### 4.2. Форматы сертификатов и каналы публикации мировых рекордов (2026)

| Репозиторий / канал | Стандартный формат файлов | Обязательные требования верификации в 2026 г. |
|---|---|---|
| **Université de Lille** ([fmm.univ-lille.fr](https://fmm.univ-lille.fr/)) & [`sedoglavic/fmm_digest`](https://github.com/sedoglavic/fmm_digest) | Maple `.tensor.mpl.bz2`, `.LRP.mpl.bz2`, `.raw.mpl.bz2`; мета-индекс `fmm_sota.json` (JSON Schema Draft 2020-12 с ORCID автора и SHA-256 `previous_hash`) | Прямая связь с A. Sedoglavic (`CRISTAL`, Univ. Lille) или PR/issue в `sedoglavic/fmm_digest`; точная проверка $AB - \sum m_r W_r^\top = 0$ (в Maple/Python exact `int`/`Fraction`), желательно указать точную область определения кольца (например, $\mathbb{Z}_T$, $\mathbb{Z}$, $\mathbb{Z}[1/2]$). |
| **Perminov Catalog** ([`dronperminov/FastMatrixMultiplication`](https://github.com/dronperminov/FastMatrixMultiplication)) | Полный `.json` (`n, m, z2, u, v, w, multiplications, elements`), редуцированный `_reduced.json` (`u_fresh, v_fresh, w_fresh`), Maple `.m`, текстовый `.exp` | PR в `dronperminov/FastMatrixMultiplication` (как делали `gsidebottom`, `arhancanli`, `khoruzhii`, `cloudroom-math` в октябре–ноябре 2026!). Приоритет отдаётся схемам в $\mathbb{Z}_T$, но принимаются любые новые рекорды ранга в $\mathbb{Z}$ / $\mathbb{Q}$. |
| **MatMulCatalog** ([`solven-eu/matmulcatalog`](https://github.com/solven-eu/matmulcatalog), `arXiv:2606.13408`) | JSON-схема с явным массивом `fields` (`["F2"]`, `["F2","F3","Z","Q","R","C"]`), флагом `zt`, SLP-блоком и provenance | Принимает **в том числе рекорды над $\mathbb{F}_2$ и $\mathbb{F}_3$**, отсутствующие в Lille! |
| **PLinOpt / arXiv / Zenodo / Lean 4** | [`jgdumas/plinopt`](https://github.com/jgdumas/plinopt) SLP-программы + препринт `cs.SC` / `cs.CC` + Zenodo DOI + (новый золотой стандарт осени 2026!) **машинный сертификат в Lean 4** | Опыт осени 2026 (`arhancanli/quantavane-fmm-records`, `gsidebottom/mat_mul`, `Tomar arXiv:2610.11823`): короткий проверяемый файл `Correct.lean` (проверка уравнений Брента точным `decide` / `native_decide` над $\mathbb{Z}$ или $\mathbb{Q}$, без `Float` и `sorryAx`) гарантирует мгновенное принятие сообществом. |

### 4.3. Стандарты аддитивной сложности и тернарного подъёма (2026)

1. **Три метрики аддитивной сложности:**
   - **Наивная (`naive` / `cn`):** $(\mathrm{nnz}(U) - R) + (\mathrm{nnz}(V) - R) + (\mathrm{nnz}(W) - mp)$ — базовая метрика при поиске и в `status.json`.
   - **Редуцированная плоская (CSE / SLP, `reduced` / `cr`):** минимизация линейных цепочек (Paar / `ternary_addition_reducer` / `PLinOpt`). Рекорды 2026: $\langle 3,3,3:23 \rangle = \mathbf{55}$ сложений (`arXiv:2607.28676` / `gsidebottom/mat_mul`), $\langle 4,4,4:48 \rangle = \mathbf{201}$ опер. над $\mathbb{Z}[1/2]$ (197 слож. + 4 деления на 2, `gsidebottom/mat_mul` Oct 2026), $\langle 4,4,4:47 \rangle = \mathbf{166}$ XOR над $\mathbb{F}_2$ (`gsidebottom/mat_mul` Nov 2026).
   - **Альтернативный базис (Karstadt–Schwartz):** перенос части сложений в $O(n^2)$-преобразование базиса; рекорд 2026 для $\langle 3,3,3:23 \rangle = \mathbf{51}$ сложение на рекурсивный шаг (`arXiv:2609.32388` и машинно-доказанная оптимальность класса в Lean 4 — `arXiv:2611.12907`).
2. **Иерархия предпочтения колец при подъёме (от лучшего к приемлемому):**
   $$\mathbb{Z}_T = \{-1, 0, 1\} \;\succ\; \mathbb{Z}\text{ (малые }|c|\le 2\text{)} \;\succ\; \mathbb{Z}[1/2]\text{ (сдвиги }2^{-k}\text{)} \;\succ\; \mathbb{Q}\text{ (малые знаменатели)} \;\succ\; \mathbb{F}_2.$$

---

## 5. Ранжированный ТОП-8 атакуемых мировых рекордов для рабочей станции (`RTX 5060 8 GB` + `12-core CPU`)

Машиночитаемая версия сохранена в [`sota_targets.json`](file:///e:/эксперименты/исследования/fmm_discovery/artifacts/sota_targets.json). Все 8 целей **гарантированно открыты на 2026-11-22**, укладываются в **одно `uint64`-слово** ($\max(mn, np, mp) \le 64$) и отобраны по произведению *(научный резонанс × вероятность прорыва на нашей машине)*.

| Ранг | Формат | Трек | `max_flat` | Текущие рекорды ($\mathbb{Z}_2 / \mathbb{Z}_T / \mathbb{Z} / \mathbb{Q}$) | Цель (кольцо : ранг) | Уверенность | Главный вектор атаки (`uint64` GPU/CPU + `SLEEP-DAG-RSI`) |
|---:|---|---|---:|---|---|---|---|
| **1** | **$\langle 4,5,5 \rangle$** | **Track A** | 25 | `73 / 76 / 76 / 76` (`Arai 2024` / `AlphaTensor 2022`) | **`ZT` (или `Z/Q`) $\le 75$** *(stretch `74–73`, $\omega < \log_2 7$!)* | **High** | **Крупнейший малый разрыв подъёма (3 ранга!).** Массовый сбор неизоморфных $\mathbb{Z}_2$-орбит рангов 75, 74, 73 на RTX 5060 с онлайн **mod-4 / mod-8 оракулом поднимаемости** + Z3/OR-Tools тернарный подъём; параллельно прямой `SLEEP-DAG-RSI` мета-флип в $\mathbb{Z}_T$ (`4x4x5:61` $\uparrow$, `5x5x5:93` $\downarrow$, `4x5x6:90` $\downarrow$) и $\mathbb{Z}_3$-флип-граф. |
| **2** | **$\langle 4,5,6 \rangle$** | **Track A** | 30 | `89 / 90 / 90 / 90` (`Kauers–Wood 2025`, `Perminov 2025` / `AlphaEvolve 2025`) | **`ZT` (или `Z/Q`) $= 89$** | **High** | **Свежий 2025–2026 разрыв в 1 ранг + гребень `+1.0`.** В `mkauers/456` заархивированы всего 2 mod-2 схемы; на `uint32` GPU генерируем тысячи новых $\mathbb{Z}_2$-бассейнов ранга 89 через `plus`-туннелирование с фильтром mod-4 + guided Hensel lifting + проекция в $\mathbb{Z}_T$ из `5x5x6:110` и `4x5x7:104`. |
| **3** | **$\langle 6,6,7 \rangle$** | **Track B** | 42 | `183 / 183 / 183 / 183` (`Kauers–Wood Oct 2025`, `Perminov ZT`) | **`ZT` (и/или `Z2`) $\le 182$** *(stretch $\le 180$)* | **High** | **Аномальный гребень `+5.0` между `6x6x6:153` и `6x6x8:203` (середина `178.0`!).** Двунаправленный `SLEEP-DAG-RSI` мета-флип (`6x6x8:203` $\downarrow$ и `6x6x6:153` $\uparrow$) + warp-level generalized multi-summand flips (`arXiv:2609.17533`) + энтропийное `plus`-туннелирование ($\tau=4.0$). |
| **4** | **$\langle 5,8,8 \rangle$** | **Track B** | 64 | `230 / 230 / 230 / 230` (чисто составная Lille `3·26 + 4·38`) | **`ZT` / `Z2` $\le 229$** *(stretch $\le 226$)* | **High** | **Гребень #1 во всём кубе $[4..8]^3$ (`+7.0` над серединой `223.0` между `4x8x8:180` и `6x8x8:266`!) и 100% составной рекорд.** Старт из мульти-сидов (блочные разбиения + проекция `6x8x8:266` + расширение `4x8x8:180`), `SLEEP-DAG-RSI` межблочный `CROSSOVER` на границе `uint64`. |
| **5** | **$\langle 7,7,7 \rangle$** | **Track A+B** | 49 | `245 / 250 / 250 / 249` (`Medley Oct 2026` / составная Lille) | **`ZT` / `Q` $\le 248$** *(stretch `245–247`)* | **High** | **Худшая $\omega=2.8354$ (`+13.23` над Штрассеном) и разрыв `245 vs 249/250`, составной рекорд в $\mathrm{char}\,0$!** Два параллельных потока: (а) прямой `ZT` мета-флип из `6x7x7:212` и `6x6x6:153` + $C_3/S_3$-симметричные прогулки; (б) GPU $\mathbb{Z}_2$-поиск от `Medley 245` в сторону поднимаемых орбит ранга `246–248` (плюс попытка $\mathbb{Z}_2 \le 244$). |
| **6** | **$\langle 5,7,8 \rangle$** | **Track B** | 56 | `204 / 204 / 204 / 204` (`Perminov 2026`, спуск `205->204`) | **`ZT` / `Z2` $\le 203$** *(stretch $\le 201$)* | **High** | **2-я худшая $\omega=2.8314$ в $[4..8]^3$ и гребень `+4.5` (`4x7x8:161`, `6x7x8:238` $\to 199.5$).** Двунаправленный мета-флип (`4x7x8` $\uparrow$, `6x7x8` $\downarrow$, `5x6x8:170` $\uparrow$) в `uint64` GPU/CPU пулах с `BackwardSubGoalDecomposer`. |
| **7** | **$\langle 4,4,8 \rangle$** | **Track A** | 32 | `94 / 96 / 96 / 95` (`Perminov 2025/2026`, `AlphaEvolve 2025`) | **`ZT` $\le 95$** *(stretch `Q/ZT = 94`)* | **Med-High** | **Двойной открытый разрыв `94 < 95 < 96`, `uint32`!** НЕ удваивать флип-изолированную $\langle 4,4,4:47 \rangle$; вместо этого спускаться проекцией с прорывного $\langle 4,4,10:115 \rangle$ (`ZT`, Perminov Mar 2026) и искать нерасщеплённые орбиты ранга 94–95 с mod-4-оракулом. |
| **8** | **$\langle 4,4,5 \rangle$** | **Track A** | 20 | `60 / 61 / 61 / 61` (`Kauers–Moosbauer 2023` / `AlphaEvolve 2025`) | **`Q` (или `ZT`) $= 60$** *($\omega = 2.8030 < \log_2 7$!)* | **Medium** | **Сверхпрестижная цель (`max_flat=20`, победа над AlphaEvolve 2025 + спуск ниже порога Штрассена!).** GPU-поиск над **$\mathbb{Z}_3$ и $\mathbb{Z}_5$** (в $100\times$ глубже CPU-пробы Kauers–Wood, чтобы поймать $\mathbb{Z}[1/2]$-схему) + mod-4-направленные $\mathbb{Z}_2$-прогулки + SAT partial-fixing вокруг `4x4x5:61 ZT`. |

### Резерв второй очереди и престижные Moonshot-цели (вне Топ-8)
- **$\langle 4,4,9 \rangle$** (`Q=104` Smirnov vs `ZT=Z2=107`, `max_flat=36`): идеальный побочный продукт проекции `4x4x10:115 ZT` $\to$ `4x4x9` при отработке цели #7 (`4x4x8`) — шанс мгновенно закрыть 3-ранговый разрыв `ZT: 107 -> <=106/104`.
- **$\langle 7,7,8 \rangle$** (`Z2=275 < Q=277 < ZT=278`, составная в $\mathrm{char}\,0$, гребень `+5.0`, `max_flat=56`): естественное расширение пайплайна цели #5 (`7x7x7`).
- **$\langle 3,3,6 \rangle$** (`Q=40` с $\pm 1/8$ vs `ZT=Z2=42`, `max_flat=18`): фундаментальная загадка `arXiv:2603.02398` Sec 7.4; стоит прогнать дешёвый SAT/Z3-зонд на `ZT=41` (уже дало бы $\omega = 2.79287 < \log_2 7$ над произвольным кольцом!).
- **$\langle 4,4,4 \rangle$** (`Z2=47 < Q=48 < ZT=49`, `max_flat=16`): главный «святой Грааль» FMM (убрать $1/2$ из схемы AlphaEvolve/DPS, получив ранг 48 над $\mathbb{Z}$ / $\mathbb{Z}_T$), но из-за доказанных барьеров жёсткости Zaru (2026) и Agarwal (2026) имеет низкую вероятность (`low`) и не должен съедать основной GPU-бюджет Фазы 1–2 до взятия целей #1–#6.
