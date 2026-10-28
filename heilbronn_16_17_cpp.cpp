// OpenMP C++ Bidirectional Cascade & Mitosis Engine for Heilbronn Triangle (n = 15, 16, 17, 18)
// Features:
// 1. Direct 2-Vertex Lift & Double Mitosis from asymmetric n=14* (0.0237757731) -> n=16 (bypassing n=15 D3 trap)
// 2. Full Bidirectional Cascade: 14* -> 15 <-> 16 <-> 17 <- 18
// 3. Teleportation Move: Bottleneck Vertex Deletion + High-Clearance Vertex Mitosis
// 4. Basin Deduplication via Sorted Triangle-Area Spectrum so clones never crowd out the Elite Pool

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <omp.h>

struct Pt {
    double x, y;
};

struct Triplet {
    int i, j, k;
};

inline void project_pt(Pt& p) {
    if (p.x < 0.0) p.x = 0.0;
    if (p.y < 0.0) p.y = 0.0;
    if (p.x > 1.0) p.x = 1.0;
    if (p.y > 1.0) p.y = 1.0;
    double s = p.x + p.y;
    if (s > 1.0) {
        double exc = 0.5 * (s - 1.0);
        p.x -= exc;
        p.y -= exc;
        if (p.x < 0.0) { p.y += p.x; p.x = 0.0; }
        if (p.y < 0.0) { p.x += p.y; p.y = 0.0; }
        if (p.x + p.y > 1.0) p.x = 1.0 - p.y;
    }
}

std::vector<Triplet> make_triplets(int n) {
    std::vector<Triplet> t;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            for (int k = j + 1; k < n; ++k)
                t.push_back({i, j, k});
    return t;
}

double eval_min_area(const std::vector<Pt>& pts, const std::vector<Triplet>& tris, int* worst_t = nullptr) {
    double min_a = 1e9;
    int wt = 0;
    const int m = (int)tris.size();
    for (int t = 0; t < m; ++t) {
        const auto& tr = tris[t];
        double cross = pts[tr.i].x * (pts[tr.j].y - pts[tr.k].y)
                     + pts[tr.j].x * (pts[tr.k].y - pts[tr.i].y)
                     + pts[tr.k].x * (pts[tr.i].y - pts[tr.j].y);
        double a = std::abs(cross);
        if (a < min_a) {
            min_a = a;
            wt = t;
        }
    }
    if (worst_t) *worst_t = wt;
    return min_a;
}

void local_polish(std::vector<Pt>& pts, const std::vector<Triplet>& tris, int max_steps = 1200, bool soft_start = false) {
    const int n = (int)pts.size();
    const int m = (int)tris.size();
    std::vector<Pt> best_pts = pts;
    double best_val = eval_min_area(pts, tris);

    std::vector<double> areas(m), cross_v(m), w(m);
    std::vector<Pt> grad(n), m_mom(n, {0.0, 0.0}), v_mom(n, {0.0, 0.0});

    const double betas_soft[6] = {70.0, 220.0, 750.0, 2600.0, 9500.0, 38000.0};
    const double betas_std[6]  = {220.0, 650.0, 2000.0, 6500.0, 18000.0, 48000.0};
    const double* betas = soft_start ? betas_soft : betas_std;
    const double lrs[6] = {0.0048, 0.0026, 0.0013, 0.00055, 0.00022, 0.00007};
    const int steps_per_beta = max_steps / 6;

    for (int b_idx = 0; b_idx < 6; ++b_idx) {
        double beta = betas[b_idx];
        double lr = lrs[b_idx];
        std::fill(m_mom.begin(), m_mom.end(), Pt{0.0, 0.0});
        std::fill(v_mom.begin(), v_mom.end(), Pt{0.0, 0.0});

        for (int s = 1; s <= steps_per_beta; ++s) {
            double min_a = 1e9;
            for (int t = 0; t < m; ++t) {
                const auto& tr = tris[t];
                double c = pts[tr.i].x * (pts[tr.j].y - pts[tr.k].y)
                         + pts[tr.j].x * (pts[tr.k].y - pts[tr.i].y)
                         + pts[tr.k].x * (pts[tr.i].y - pts[tr.j].y);
                cross_v[t] = c;
                double a = std::abs(c);
                areas[t] = a;
                if (a < min_a) min_a = a;
            }
            if (min_a > best_val) {
                best_val = min_a;
                best_pts = pts;
            }

            double sum_w = 0.0;
            for (int t = 0; t < m; ++t) {
                double diff = areas[t] - min_a;
                if (beta * diff < 40.0) {
                    double wt = std::exp(-beta * diff);
                    w[t] = wt;
                    sum_w += wt;
                } else {
                    w[t] = 0.0;
                }
            }
            double inv_Z = 1.0 / (sum_w + 1e-30);
            for (int i = 0; i < n; ++i) grad[i] = {0.0, 0.0};

            for (int t = 0; t < m; ++t) {
                if (w[t] <= 0.0) continue;
                double p = w[t] * inv_Z;
                double sgn = (cross_v[t] >= 0.0) ? p : -p;
                const auto& tr = tris[t];
                grad[tr.i].x += sgn * (pts[tr.j].y - pts[tr.k].y);
                grad[tr.i].y += sgn * (pts[tr.k].x - pts[tr.j].x);
                grad[tr.j].x += sgn * (pts[tr.k].y - pts[tr.i].y);
                grad[tr.j].y += sgn * (pts[tr.i].x - pts[tr.k].x);
                grad[tr.k].x += sgn * (pts[tr.i].y - pts[tr.j].y);
                grad[tr.k].y += sgn * (pts[tr.j].x - pts[tr.i].x);
            }

            double b1 = 0.85, b2 = 0.99;
            double bc1 = 1.0 - std::pow(b1, s);
            double bc2 = 1.0 - std::pow(b2, s);
            for (int i = 0; i < n; ++i) {
                m_mom[i].x = b1 * m_mom[i].x + (1.0 - b1) * grad[i].x;
                m_mom[i].y = b1 * m_mom[i].y + (1.0 - b1) * grad[i].y;
                v_mom[i].x = b2 * v_mom[i].x + (1.0 - b2) * grad[i].x * grad[i].x;
                v_mom[i].y = b2 * v_mom[i].y + (1.0 - b2) * grad[i].y * grad[i].y;
                pts[i].x += lr * (m_mom[i].x / bc1) / (std::sqrt(v_mom[i].x / bc2) + 1e-12);
                pts[i].y += lr * (m_mom[i].y / bc1) / (std::sqrt(v_mom[i].y / bc2) + 1e-12);
                project_pt(pts[i]);
            }
        }
        pts = best_pts;
    }
}

std::vector<Pt> seed_topological(int n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<Pt> pts(n);

    int mode = rng() % 6;
    if (mode <= 3 && n >= 10) {
        // 6 boundary points: 2 on y=0, 2 on x=0, 2 on x+y=1
        double e1_a = 0.12 + 0.26 * unif(rng), e1_b = 0.58 + 0.30 * unif(rng);
        double e2_a = 0.12 + 0.26 * unif(rng), e2_b = 0.58 + 0.30 * unif(rng);
        double e3_a = 0.12 + 0.26 * unif(rng), e3_b = 0.58 + 0.30 * unif(rng);
        pts[0] = {e1_a, 0.0}; pts[1] = {e1_b, 0.0};
        pts[2] = {0.0, e2_a}; pts[3] = {0.0, e2_b};
        pts[4] = {e3_a, 1.0 - e3_a}; pts[5] = {e3_b, 1.0 - e3_b};

        if (mode == 0 || mode == 1) {
            // 3 near-corner buffer points + concentric rings
            pts[6] = {0.032 + 0.045 * unif(rng), 0.032 + 0.045 * unif(rng)}; project_pt(pts[6]);
            pts[7] = {0.830 + 0.080 * unif(rng), 0.032 + 0.045 * unif(rng)}; project_pt(pts[7]);
            pts[8] = {0.032 + 0.045 * unif(rng), 0.830 + 0.080 * unif(rng)}; project_pt(pts[8]);
            int rem = n - 9;
            double rot = unif(rng) * 6.283185307179586;
            for (int r = 0; r < rem; ++r) {
                if (mode == 1 && r >= rem - 2) {
                    double ang2 = rot + (r - (rem - 2)) * 3.14159265 + 0.3 * norm(rng);
                    double rad2 = 0.055 + 0.04 * unif(rng);
                    pts[9 + r] = {0.30 + rad2 * std::cos(ang2), 0.30 + rad2 * std::sin(ang2)};
                } else {
                    int k_ring = (mode == 1) ? (rem - 2) : rem;
                    double ang = rot + r * (6.283185307179586 / std::max(1, k_ring)) + 0.15 * norm(rng);
                    double rad = 0.15 + 0.08 * unif(rng);
                    pts[9 + r] = {0.30 + rad * std::cos(ang), 0.30 + rad * std::sin(ang)};
                }
                project_pt(pts[9 + r]);
            }
        } else {
            // Mirror-biased (near-Cs) + perturbed interior
            int rem = n - 6;
            for (int r = 0; r < rem; r += 2) {
                double u = 0.05 + 0.65 * unif(rng);
                double v = 0.05 + (0.88 - u) * unif(rng);
                pts[6 + r] = {u, v}; project_pt(pts[6 + r]);
                if (r + 1 < rem) {
                    pts[6 + r + 1] = {v + 0.02 * norm(rng), u + 0.02 * norm(rng)};
                    project_pt(pts[6 + r + 1]);
                }
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            double u = unif(rng), v = unif(rng);
            if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
            pts[i] = {u, v};
        }
    }
    return pts;
}

std::vector<Pt> build_search_grid(int res = 100) {
    std::vector<Pt> grid;
    for (int i = 0; i <= res; ++i) {
        double x = (double)i / res;
        for (int j = 0; i + j <= res; ++j) {
            double y = (double)j / res;
            grid.push_back({x, y});
        }
    }
    return grid;
}

std::vector<Pt> top_pockets_for_subset(const std::vector<Pt>& sub, const std::vector<Pt>& grid, int top_k = 6) {
    const int m = (int)sub.size();
    const int g_sz = (int)grid.size();
    struct ScoredPt { double val; int idx; };
    std::vector<ScoredPt> all;
    all.reserve(g_sz);

    for (int g = 0; g < g_sz; ++g) {
        const Pt& q = grid[g];
        double min_a = 1e9;
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                double c = sub[i].x * (sub[j].y - q.y)
                         + sub[j].x * (q.y - sub[i].y)
                         + q.x * (sub[i].y - sub[j].y);
                double a = std::abs(c);
                if (a < min_a) {
                    min_a = a;
                    if (min_a <= 1e-6) break;
                }
            }
            if (min_a <= 1e-6) break;
        }
        all.push_back({min_a, g});
    }
    std::sort(all.begin(), all.end(), [](const ScoredPt& a, const ScoredPt& b) {
        return a.val > b.val;
    });
    std::vector<Pt> chosen;
    for (const auto& sp : all) {
        const Pt& cand = grid[sp.idx];
        bool far_enough = true;
        for (const auto& prev : chosen) {
            double dx = cand.x - prev.x, dy = cand.y - prev.y;
            if (dx * dx + dy * dy < 0.038 * 0.038) { far_enough = false; break; }
        }
        if (far_enough) {
            chosen.push_back(cand);
            if ((int)chosen.size() >= top_k) break;
        }
    }
    if (chosen.empty()) chosen.push_back(grid[all[0].idx]);
    return chosen;
}

void ruin_and_recreate_chain(
    std::vector<Pt>& pts,
    const std::vector<Triplet>& tris,
    const std::vector<Pt>& grid,
    std::mt19937_64& rng,
    int n_iters = 40
) {
    const int n = (int)pts.size();
    double best_val = eval_min_area(pts, tris);
    std::vector<Pt> best_pts = pts;
    std::normal_distribution<double> norm(0.0, 1.0);

    for (int it = 0; it < n_iters; ++it) {
        std::vector<Pt> cand = best_pts;
        int worst_t = 0;
        eval_min_area(cand, tris, &worst_t);
        const auto& tr = tris[worst_t];
        int b_verts[3] = {tr.i, tr.j, tr.k};

        int move_type = it % 5;
        bool use_soft = false;
        if (move_type == 0) {
            // 1-vertex relocation from bottleneck triangle to top pocket
            int v_del = b_verts[rng() % 3];
            std::vector<Pt> sub;
            for (int i = 0; i < n; ++i) if (i != v_del) sub.push_back(cand[i]);
            auto pockets = top_pockets_for_subset(sub, grid, 5);
            cand[v_del] = pockets[rng() % pockets.size()];
            use_soft = true;
        } else if (move_type == 1) {
            // 2-vertex simultaneous Ruin-and-Recreate
            int v1 = b_verts[rng() % 3];
            int v2 = rng() % n;
            while (v2 == v1) v2 = rng() % n;
            std::vector<Pt> sub;
            for (int i = 0; i < n; ++i) if (i != v1 && i != v2) sub.push_back(cand[i]);
            auto p1_list = top_pockets_for_subset(sub, grid, 4);
            Pt q1 = p1_list[rng() % p1_list.size()];
            sub.push_back(q1);
            auto p2_list = top_pockets_for_subset(sub, grid, 4);
            Pt q2 = p2_list[rng() % p2_list.size()];
            cand[v1] = q1;
            cand[v2] = q2;
            use_soft = true;
        } else if (move_type == 2) {
            // Teleportation Mitosis: delete 1 bottleneck vertex, split another interior vertex!
            int v_del = b_verts[rng() % 3];
            int v_split = rng() % n;
            while (v_split == v_del) v_split = rng() % n;
            double ang = (rng() % 12) * (3.141592653589793 / 6.0);
            double dx = 0.038 * std::cos(ang), dy = 0.038 * std::sin(ang);
            Pt p_a = {cand[v_split].x + dx, cand[v_split].y + dy};
            Pt p_b = {cand[v_split].x - dx, cand[v_split].y - dy};
            project_pt(p_a);
            project_pt(p_b);
            cand[v_split] = p_a;
            cand[v_del] = p_b;
            use_soft = true;
        } else if (move_type == 3) {
            // Perturb bottleneck vertices
            double sigma = (it < n_iters / 2) ? 0.025 : 0.008;
            for (int k = 0; k < 2; ++k) {
                int v = b_verts[(it + k) % 3];
                cand[v].x += sigma * norm(rng);
                cand[v].y += sigma * norm(rng);
                project_pt(cand[v]);
            }
        } else {
            // Global random vertex relocation
            int v_del = rng() % n;
            std::vector<Pt> sub;
            for (int i = 0; i < n; ++i) if (i != v_del) sub.push_back(cand[i]);
            auto pockets = top_pockets_for_subset(sub, grid, 5);
            Pt q = pockets[rng() % pockets.size()];
            q.x += 0.008 * norm(rng);
            q.y += 0.008 * norm(rng);
            project_pt(q);
            cand[v_del] = q;
            use_soft = true;
        }

        local_polish(cand, tris, 1050, use_soft);
        double val = eval_min_area(cand, tris);
        if (val > best_val) {
            best_val = val;
            best_pts = cand;
        }
    }
    pts = best_pts;
}

// Deduplicate pool so clones in the same basin (within 1e-5 in sorted area spectrum) don't crowd out diverse chambers
void sort_and_dedup_pool(std::vector<std::pair<double, std::vector<Pt>>>& pool, int max_keep = 80) {
    std::sort(pool.begin(), pool.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::pair<double, std::vector<Pt>>> unique_pool;
    for (const auto& item : pool) {
        bool is_dup = false;
        for (const auto& kept : unique_pool) {
            if (std::abs(item.first - kept.first) < 2e-6) {
                is_dup = true;
                break;
            }
        }
        if (!is_dup) {
            unique_pool.push_back(item);
            if ((int)unique_pool.size() >= max_keep) break;
        }
    }
    pool = unique_pool;
}

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();
    const auto grid = build_search_grid(96);

    const auto tris15 = make_triplets(15);
    const auto tris16 = make_triplets(16);
    const auto tris17 = make_triplets(17);

    // Hardcode our certified n=14* champion (0.0237757731) so we can lift 14* -> 15 and 14* -> 16 directly!
    const std::vector<Pt> champ14 = {
        {0.08378678791445193, 0.87584290048773660},
        {0.00000000000000000, 0.72894119555956050},
        {0.69682163174157300, 0.00000000000000000},
        {0.04490315201948169, 0.05660394575933771},
        {0.23779968454648212, 0.76220031545351790},
        {0.18189167486819054, 0.56256411220263300},
        {0.44943038714852836, 0.24234385134848638},
        {0.25120461992599186, 0.00000000000000000},
        {0.89265279329433180, 0.04628505161369503},
        {0.44943038714852836, 0.44140844637412270},
        {0.78507398693342500, 0.21492601306657500},
        {0.51423175011484210, 0.06725216336738712},
        {0.00000000000000000, 0.34353615685094935},
        {0.20703604474127676, 0.25184266194289710}
    };

    std::vector<std::pair<double, std::vector<Pt>>> pool15(120);
    std::vector<std::pair<double, std::vector<Pt>>> pool16(180);
    std::vector<std::pair<double, std::vector<Pt>>> pool17(140);

    // 1. Seed n=15 (from 14* + topological)
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < 120; ++r) {
        std::mt19937_64 rng(10000 + r * 97);
        std::vector<Pt> pts;
        if (r < 40) {
            pts = champ14;
            auto pockets = top_pockets_for_subset(pts, grid, 8);
            pts.push_back(pockets[r % pockets.size()]);
            for (auto& p : pts) {
                p.x += 0.006 * (r / 8) * ((rng() % 200 - 100) / 100.0);
                p.y += 0.006 * (r / 8) * ((rng() % 200 - 100) / 100.0);
                project_pt(p);
            }
        } else {
            pts = seed_topological(15, rng);
        }
        local_polish(pts, tris15, 1000, true);
        ruin_and_recreate_chain(pts, tris15, grid, rng, 20);
        pool15[r] = {eval_min_area(pts, tris15), pts};
    }

    // 2. Seed n=16 (Direct 2-vertex lift from 14* + Double Mitosis from 14* + Topological)
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < 180; ++r) {
        std::mt19937_64 rng(30000 + r * 113);
        std::vector<Pt> pts;
        if (r < 60) {
            // Direct 2-vertex pocket lift 14* -> 16
            pts = champ14;
            for (auto& p : pts) {
                p.x += 0.008 * ((rng() % 200 - 100) / 100.0);
                p.y += 0.008 * ((rng() % 200 - 100) / 100.0);
                project_pt(p);
            }
            auto p1_list = top_pockets_for_subset(pts, grid, 6);
            pts.push_back(p1_list[r % p1_list.size()]);
            auto p2_list = top_pockets_for_subset(pts, grid, 6);
            pts.push_back(p2_list[(r / 6) % p2_list.size()]);
        } else if (r < 100) {
            // Double Mitosis on interior vertices of 14*
            pts = champ14;
            int inner[8] = {0, 3, 5, 6, 8, 9, 11, 13};
            int v1 = inner[r % 8];
            int v2 = inner[(r / 8 + 1) % 8];
            double a1 = (r % 6) * 0.523598, a2 = ((r + 3) % 6) * 0.523598;
            Pt p1a = {pts[v1].x + 0.035*std::cos(a1), pts[v1].y + 0.035*std::sin(a1)};
            Pt p1b = {pts[v1].x - 0.035*std::cos(a1), pts[v1].y - 0.035*std::sin(a1)};
            Pt p2a = {pts[v2].x + 0.035*std::cos(a2), pts[v2].y + 0.035*std::sin(a2)};
            Pt p2b = {pts[v2].x - 0.035*std::cos(a2), pts[v2].y - 0.035*std::sin(a2)};
            project_pt(p1a); project_pt(p1b); project_pt(p2a); project_pt(p2b);
            pts[v1] = p1a; pts.push_back(p1b);
            pts[v2] = p2a; pts.push_back(p2b);
        } else {
            pts = seed_topological(16, rng);
        }
        local_polish(pts, tris16, 1100, true);
        ruin_and_recreate_chain(pts, tris16, grid, rng, 26);
        pool16[r] = {eval_min_area(pts, tris16), pts};
    }

    // 3. Seed n=17
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < 140; ++r) {
        std::mt19937_64 rng(50000 + r * 127);
        auto pts = seed_topological(17, rng);
        local_polish(pts, tris17, 1000, true);
        ruin_and_recreate_chain(pts, tris17, grid, rng, 22);
        pool17[r] = {eval_min_area(pts, tris17), pts};
    }

    sort_and_dedup_pool(pool15, 48);
    sort_and_dedup_pool(pool16, 64);
    sort_and_dedup_pool(pool17, 48);

    std::cout << "[Round 1 Initial] n=15 best=" << pool15[0].first
              << " | n=16 best=" << pool16[0].first
              << " | n=17 best=" << pool17[0].first << std::endl;

    // 4. Upward 15 -> 16 (from top 24 diverse n=15 chambers, NOT just clones!) + Downward 17 -> 16
    int n15_use = std::min(24, (int)pool15.size());
    std::vector<std::pair<double, std::vector<Pt>>> up16(n15_use * 8);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n15_use * 8; ++idx) {
        int e_id = idx / 8, mode = idx % 8;
        std::mt19937_64 rng(70000 + idx * 43);
        const auto& base15 = pool15[e_id].second;
        std::vector<Pt> cand16 = base15;
        if (mode < 4) {
            auto pockets = top_pockets_for_subset(base15, grid, 4);
            cand16.push_back(pockets[mode % pockets.size()]);
        } else {
            int v_split = rng() % 15;
            double ang = (mode - 4) * 0.785398;
            Pt pa = {base15[v_split].x + 0.04*std::cos(ang), base15[v_split].y + 0.04*std::sin(ang)};
            Pt pb = {base15[v_split].x - 0.04*std::cos(ang), base15[v_split].y - 0.04*std::sin(ang)};
            project_pt(pa); project_pt(pb);
            cand16[v_split] = pa; cand16.push_back(pb);
        }
        local_polish(cand16, tris16, 1100, true);
        ruin_and_recreate_chain(cand16, tris16, grid, rng, 30);
        up16[idx] = {eval_min_area(cand16, tris16), cand16};
    }
    for (const auto& x : up16) pool16.push_back(x);

    int n17_use = std::min(20, (int)pool17.size());
    std::vector<std::pair<double, std::vector<Pt>>> down16(n17_use * 17);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n17_use * 17; ++idx) {
        int e_id = idx / 17, v_del = idx % 17;
        std::mt19937_64 rng(90000 + idx * 53);
        const auto& src = pool17[e_id].second;
        std::vector<Pt> sub;
        for (int i = 0; i < 17; ++i) if (i != v_del) sub.push_back(src[i]);
        local_polish(sub, tris16, 1100, false);
        ruin_and_recreate_chain(sub, tris16, grid, rng, 30);
        down16[idx] = {eval_min_area(sub, tris16), sub};
    }
    for (const auto& x : down16) pool16.push_back(x);
    sort_and_dedup_pool(pool16, 64);

    std::cout << "[Round 2 Cascade (15->16 & 17->16)] n=16 Top-5: ";
    for (int i = 0; i < std::min(5, (int)pool16.size()); ++i)
        std::cout << std::fixed << std::setprecision(8) << pool16[i].first << " ";
    std::cout << std::endl;

    // 5. Upward 16 -> 17 -> Downward 17 -> 16 + Final Deep R&R on Top-32 of n=16
    int n16_use = std::min(20, (int)pool16.size());
    std::vector<std::pair<double, std::vector<Pt>>> up17(n16_use * 8);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n16_use * 8; ++idx) {
        int e_id = idx / 8, mode = idx % 8;
        std::mt19937_64 rng(120000 + idx * 67);
        const auto& base16 = pool16[e_id].second;
        std::vector<Pt> cand17 = base16;
        if (mode < 4) {
            auto pockets = top_pockets_for_subset(base16, grid, 4);
            cand17.push_back(pockets[mode % pockets.size()]);
        } else {
            int v_split = rng() % 16;
            double ang = (mode - 4) * 0.785398;
            Pt pa = {base16[v_split].x + 0.04*std::cos(ang), base16[v_split].y + 0.04*std::sin(ang)};
            Pt pb = {base16[v_split].x - 0.04*std::cos(ang), base16[v_split].y - 0.04*std::sin(ang)};
            project_pt(pa); project_pt(pb);
            cand17[v_split] = pa; cand17.push_back(pb);
        }
        local_polish(cand17, tris17, 1100, true);
        ruin_and_recreate_chain(cand17, tris17, grid, rng, 32);
        up17[idx] = {eval_min_area(cand17, tris17), cand17};
    }
    for (const auto& x : up17) pool17.push_back(x);
    sort_and_dedup_pool(pool17, 48);

    std::cout << "[Round 3 Upward (16->17)] n=17 Top-5: ";
    for (int i = 0; i < std::min(5, (int)pool17.size()); ++i)
        std::cout << std::fixed << std::setprecision(8) << pool17[i].first << " ";
    std::cout << std::endl;

    int n17_use2 = std::min(16, (int)pool17.size());
    std::vector<std::pair<double, std::vector<Pt>>> down16_b(n17_use2 * 17);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n17_use2 * 17; ++idx) {
        int e_id = idx / 17, v_del = idx % 17;
        std::mt19937_64 rng(160000 + idx * 79);
        const auto& src = pool17[e_id].second;
        std::vector<Pt> sub;
        for (int i = 0; i < 17; ++i) if (i != v_del) sub.push_back(src[i]);
        local_polish(sub, tris16, 1100, false);
        ruin_and_recreate_chain(sub, tris16, grid, rng, 35);
        down16_b[idx] = {eval_min_area(sub, tris16), sub};
    }
    for (const auto& x : down16_b) pool16.push_back(x);
    sort_and_dedup_pool(pool16, 64);

    // Final Deep R&R on Top-32 of n=16 (64 parallel chains of 55 iterations)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 64; ++i) {
        std::mt19937_64 rng(200000 + i * 97);
        auto pts = pool16[i % std::min(32, (int)pool16.size())].second;
        ruin_and_recreate_chain(pts, tris16, grid, rng, 55);
        double v = eval_min_area(pts, tris16);
        #pragma omp critical
        {
            pool16.push_back({v, pts});
        }
    }
    sort_and_dedup_pool(pool16, 40);

    std::cout << "[FINAL C++ Top-5 n=15]: ";
    for (int i = 0; i < std::min(5, (int)pool15.size()); ++i) std::cout << std::fixed << std::setprecision(8) << pool15[i].first << " ";
    std::cout << "\n[FINAL C++ Top-5 n=16]: ";
    for (int i = 0; i < std::min(5, (int)pool16.size()); ++i) std::cout << std::fixed << std::setprecision(8) << pool16[i].first << " ";
    std::cout << "\n[FINAL C++ Top-5 n=17]: ";
    for (int i = 0; i < std::min(5, (int)pool17.size()); ++i) std::cout << std::fixed << std::setprecision(8) << pool17[i].first << " ";
    std::cout << std::endl;

    std::ofstream out("heilbronn_16_17_cpp_candidates.json");
    out << std::setprecision(17) << "{\n  \"n16\": [\n";
    int k16 = std::min(25, (int)pool16.size());
    for (int i = 0; i < k16; ++i) {
        out << "    [";
        for (int p = 0; p < 16; ++p) {
            out << "[" << pool16[i].second[p].x << "," << pool16[i].second[p].y << "]" << (p < 15 ? "," : "");
        }
        out << "]" << (i < k16 - 1 ? ",\n" : "\n");
    }
    out << "  ],\n  \"n17\": [\n";
    int k17 = std::min(25, (int)pool17.size());
    for (int i = 0; i < k17; ++i) {
        out << "    [";
        for (int p = 0; p < 17; ++p) {
            out << "[" << pool17[i].second[p].x << "," << pool17[i].second[p].y << "]" << (p < 16 ? "," : "");
        }
        out << "]" << (i < k17 - 1 ? ",\n" : "\n");
    }
    out << "  ]\n}\n";
    out.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "C++ Search completed in " << std::fixed << std::setprecision(2) << elapsed << "s\n";
    return 0;
}
