// Focused OpenMP C++ Storm 2:
// 1. Breaking the 15+1 Floater Lock on n=16 (starting from 0.01835554635546 where vertex 13 is a floater)
//    by deleting 1 or 2 frame vertices while forcing floater vertex 13 into the active frame + Vertex Mitosis
// 2. Lifting 16* (0.01835554635546) -> 17 via Pocket Insertion & Mitosis + 2-Vertex R&R on n=17
// 3. Downward Cascade 17 -> 16 from newly discovered n=17 chambers

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

    const double betas_soft[6] = {80.0, 250.0, 850.0, 3000.0, 11000.0, 42000.0};
    const double betas_std[6]  = {250.0, 750.0, 2400.0, 7500.0, 22000.0, 55000.0};
    const double* betas = soft_start ? betas_soft : betas_std;
    const double lrs[6] = {0.0045, 0.0024, 0.0012, 0.0005, 0.0002, 0.00006};
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

std::vector<Pt> build_search_grid(int res = 105) {
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
            if (dx * dx + dy * dy < 0.036 * 0.036) { far_enough = false; break; }
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
    int n_iters = 45
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
            int v_del = b_verts[rng() % 3];
            std::vector<Pt> sub;
            for (int i = 0; i < n; ++i) if (i != v_del) sub.push_back(cand[i]);
            auto pockets = top_pockets_for_subset(sub, grid, 5);
            cand[v_del] = pockets[rng() % pockets.size()];
            use_soft = true;
        } else if (move_type == 1) {
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
            int v_del = b_verts[rng() % 3];
            int v_split = rng() % n;
            while (v_split == v_del) v_split = rng() % n;
            double ang = (rng() % 12) * (3.141592653589793 / 6.0);
            double dx = 0.036 * std::cos(ang), dy = 0.036 * std::sin(ang);
            Pt p_a = {cand[v_split].x + dx, cand[v_split].y + dy};
            Pt p_b = {cand[v_split].x - dx, cand[v_split].y - dy};
            project_pt(p_a);
            project_pt(p_b);
            cand[v_split] = p_a;
            cand[v_del] = p_b;
            use_soft = true;
        } else if (move_type == 3) {
            double sigma = (it < n_iters / 2) ? 0.022 : 0.007;
            for (int k = 0; k < 2; ++k) {
                int v = b_verts[(it + k) % 3];
                cand[v].x += sigma * norm(rng);
                cand[v].y += sigma * norm(rng);
                project_pt(cand[v]);
            }
        } else {
            int v_del = rng() % n;
            std::vector<Pt> sub;
            for (int i = 0; i < n; ++i) if (i != v_del) sub.push_back(cand[i]);
            auto pockets = top_pockets_for_subset(sub, grid, 5);
            Pt q = pockets[rng() % pockets.size()];
            q.x += 0.007 * norm(rng);
            q.y += 0.007 * norm(rng);
            project_pt(q);
            cand[v_del] = q;
            use_soft = true;
        }

        local_polish(cand, tris, 1100, use_soft);
        double val = eval_min_area(cand, tris);
        if (val > best_val) {
            best_val = val;
            best_pts = cand;
        }
    }
    pts = best_pts;
}

void sort_and_dedup_pool(std::vector<std::pair<double, std::vector<Pt>>>& pool, int max_keep = 64) {
    std::sort(pool.begin(), pool.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::pair<double, std::vector<Pt>>> unique_pool;
    for (const auto& item : pool) {
        bool is_dup = false;
        for (const auto& kept : unique_pool) {
            if (std::abs(item.first - kept.first) < 1e-6) {
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
    const auto grid = build_search_grid(100);
    const auto tris16 = make_triplets(16);
    const auto tris17 = make_triplets(17);

    // Load our newly discovered n=16 champion (0.01835554635546114)
    const std::vector<Pt> champ16 = {
        {0.17741295157276815, 0.16622882269937692}, // pt 0
        {0.93624211908220134, 0.03071440625498531}, // pt 1
        {0.00000000000000000, 0.11600930640558861}, // pt 2
        {0.00000000000000000, 0.71311004103697362}, // pt 3
        {0.03074112170837046, 0.96925887829162954}, // pt 4
        {0.77745301502286646, 0.22254698497713354}, // pt 5
        {0.14063964267864731, 0.00000000000000000}, // pt 6
        {0.35206971767784895, 0.03659299190859004}, // pt 7
        {0.45392498378635088, 0.14103761873899712}, // pt 8
        {0.47265699413956530, 0.38821168729409200}, // pt 9
        {0.38074526099547634, 0.58621126434171000}, // pt 10
        {0.86598067864126349, 0.00000000000000000}, // pt 11
        {0.13999495731837436, 0.49623984394639098}, // pt 12
        {0.03457067749392794, 0.37871538258074805}, // pt 13: FLOATER!
        {0.17668442403077386, 0.72700531310634187}, // pt 14
        {0.32697461665108934, 0.34314826093101913}  // pt 15
    };

    // 1. Lift 16* -> 17 (160 parallel chains: pocket insertions + vertex mitosis of all 16 vertices)
    std::vector<std::pair<double, std::vector<Pt>>> pool17(160);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < 160; ++idx) {
        std::mt19937_64 rng(50000 + idx * 89);
        std::vector<Pt> cand17 = champ16;
        int mode = idx % 24;
        if (mode < 8) {
            auto pockets = top_pockets_for_subset(cand17, grid, 8);
            cand17.push_back(pockets[mode % pockets.size()]);
        } else {
            int v_split = (mode - 8) % 16;
            double ang = (idx / 24) * 0.5235987756;
            Pt pa = {cand17[v_split].x + 0.035*std::cos(ang), cand17[v_split].y + 0.035*std::sin(ang)};
            Pt pb = {cand17[v_split].x - 0.035*std::cos(ang), cand17[v_split].y - 0.035*std::sin(ang)};
            project_pt(pa); project_pt(pb);
            cand17[v_split] = pa;
            cand17.push_back(pb);
        }
        for (auto& p : cand17) {
            p.x += 0.004 * (idx / 24) * ((rng() % 200 - 100) / 100.0);
            p.y += 0.004 * (idx / 24) * ((rng() % 200 - 100) / 100.0);
            project_pt(p);
        }
        local_polish(cand17, tris17, 1100, true);
        ruin_and_recreate_chain(cand17, tris17, grid, rng, 42);
        pool17[idx] = {eval_min_area(cand17, tris17), cand17};
    }
    sort_and_dedup_pool(pool17, 48);
    std::cout << "[Stage 1 | Lifted 16* -> 17] n=17 Top-5: ";
    for (int i = 0; i < std::min(5, (int)pool17.size()); ++i)
        std::cout << std::fixed << std::setprecision(8) << pool17[i].first << " ";
    std::cout << std::endl;

    // 2. Attack on n=16:
    //    (a) Downward 17 -> 16 from the new n=17 elites
    //    (b) Floater-Lock Breaker on champ16: keep floater 13, delete 1 or 2 frame vertices, and split/insert!
    int n17_use = std::min(16, (int)pool17.size());
    std::vector<std::pair<double, std::vector<Pt>>> pool16(n17_use * 17 + 120);
    pool16[0] = {eval_min_area(champ16, tris16), champ16};

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n17_use * 17; ++idx) {
        int e_id = idx / 17, v_del = idx % 17;
        std::mt19937_64 rng(90000 + idx * 59);
        const auto& src = pool17[e_id].second;
        std::vector<Pt> sub;
        for (int i = 0; i < 17; ++i) if (i != v_del) sub.push_back(src[i]);
        local_polish(sub, tris16, 1100, false);
        ruin_and_recreate_chain(sub, tris16, grid, rng, 38);
        pool16[1 + idx] = {eval_min_area(sub, tris16), sub};
    }

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < 119; ++idx) {
        std::mt19937_64 rng(150000 + idx * 73);
        std::vector<Pt> cand = champ16;
        // Delete a frame vertex != 13 and either split vertex 13 or relocate to pocket!
        int v_frame = idx % 15;
        if (v_frame >= 13) v_frame += 1;
        if (idx % 2 == 0) {
            // Split floater vertex 13 or another interior vertex into 2!
            int v_sp = (idx % 4 == 0) ? 13 : ((idx / 4) % 16);
            if (v_sp == v_frame) v_sp = 13;
            double ang = (idx % 12) * (3.141592653589793 / 6.0);
            Pt pa = {cand[v_sp].x + 0.04*std::cos(ang), cand[v_sp].y + 0.04*std::sin(ang)};
            Pt pb = {cand[v_sp].x - 0.04*std::cos(ang), cand[v_sp].y - 0.04*std::sin(ang)};
            project_pt(pa); project_pt(pb);
            cand[v_sp] = pa;
            cand[v_frame] = pb;
        } else {
            // 2-frame-vertex Ruin-and-Recreate while keeping floater 13
            int v2 = (v_frame + 1 + (idx / 15)) % 16;
            if (v2 == 13) v2 = (v2 + 1) % 16;
            std::vector<Pt> sub;
            for (int i = 0; i < 16; ++i) if (i != v_frame && i != v2) sub.push_back(cand[i]);
            auto p1_list = top_pockets_for_subset(sub, grid, 5);
            Pt q1 = p1_list[rng() % p1_list.size()];
            sub.push_back(q1);
            auto p2_list = top_pockets_for_subset(sub, grid, 5);
            Pt q2 = p2_list[rng() % p2_list.size()];
            cand[v_frame] = q1;
            cand[v2] = q2;
        }
        local_polish(cand, tris16, 1100, true);
        ruin_and_recreate_chain(cand, tris16, grid, rng, 45);
        pool16[1 + n17_use * 17 + idx] = {eval_min_area(cand, tris16), cand};
    }

    sort_and_dedup_pool(pool16, 40);
    std::cout << "[FINAL Storm 2 Top-5 n=16]: ";
    for (int i = 0; i < std::min(5, (int)pool16.size()); ++i)
        std::cout << std::fixed << std::setprecision(8) << pool16[i].first << " ";
    std::cout << "\n[FINAL Storm 2 Top-5 n=17]: ";
    for (int i = 0; i < std::min(5, (int)pool17.size()); ++i)
        std::cout << std::fixed << std::setprecision(8) << pool17[i].first << " ";
    std::cout << std::endl;

    std::ofstream out("heilbronn_16_17_storm2_candidates.json");
    out << std::setprecision(17) << "{\n  \"n16\": [\n";
    int k16 = std::min(20, (int)pool16.size());
    for (int i = 0; i < k16; ++i) {
        out << "    [";
        for (int p = 0; p < 16; ++p) {
            out << "[" << pool16[i].second[p].x << "," << pool16[i].second[p].y << "]" << (p < 15 ? "," : "");
        }
        out << "]" << (i < k16 - 1 ? ",\n" : "\n");
    }
    out << "  ],\n  \"n17\": [\n";
    int k17 = std::min(20, (int)pool17.size());
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
    std::cout << "Storm 2 completed in " << std::fixed << std::setprecision(2) << elapsed << "s\n";
    return 0;
}
