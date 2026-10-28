// Upgraded OpenMP C++ Discovery Engine for Heilbronn Triangle Problem (n = 13 and n = 14)
// Includes:
// 1. 6-Boundary-Edge (2 per edge, 0 corners) + Interior Ring + Corner-Buffer topological seeding
// 2. Soft-to-Sharp LogSumExp Homotopy (beta = 50 -> 150 -> 500 -> 2000 -> 8000 -> 35000)
// 3. 1-Vertex and 2-Vertex Ruin-and-Recreate (14 -> 12 -> 13 -> 14 pocket reconstruction)
// 4. Vertex Mitosis (13 -> 14 interior point splitting across angles)
// 5. Bidirectional Vertical Seeding Cascade: 13 <-> 14 <-> 15

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

    const double betas_soft[6] = {60.0, 180.0, 600.0, 2200.0, 8000.0, 32000.0};
    const double betas_std[6]  = {180.0, 500.0, 1600.0, 5000.0, 15000.0, 40000.0};
    const double* betas = soft_start ? betas_soft : betas_std;
    const double lrs[6] = {0.005, 0.0028, 0.0014, 0.0006, 0.00025, 0.00008};
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

    int mode = rng() % 5;
    if (mode <= 3 && n >= 9) {
        // 6 boundary points: 2 on y=0, 2 on x=0, 2 on x+y=1
        double e1_a = 0.14 + 0.25 * unif(rng);
        double e1_b = 0.60 + 0.28 * unif(rng);
        double e2_a = 0.14 + 0.25 * unif(rng);
        double e2_b = 0.60 + 0.28 * unif(rng);
        double e3_a = 0.14 + 0.25 * unif(rng);
        double e3_b = 0.60 + 0.28 * unif(rng);
        pts[0] = {e1_a, 0.0};
        pts[1] = {e1_b, 0.0};
        pts[2] = {0.0, e2_a};
        pts[3] = {0.0, e2_b};
        pts[4] = {e3_a, 1.0 - e3_a};
        pts[5] = {e3_b, 1.0 - e3_b};

        if (mode == 0 || mode == 1) {
            // 3 near-corner buffer points (like in our 0.0265565 n=13 champion!) + (n-9) inner core points
            double c0_x = 0.035 + 0.045 * unif(rng), c0_y = 0.035 + 0.045 * unif(rng);
            double c1_x = 0.82 + 0.08 * unif(rng),   c1_y = 0.035 + 0.045 * unif(rng);
            double c2_x = 0.035 + 0.045 * unif(rng), c2_y = 0.82 + 0.08 * unif(rng);
            pts[6] = {c0_x, c0_y}; project_pt(pts[6]);
            pts[7] = {c1_x, c1_y}; project_pt(pts[7]);
            pts[8] = {c2_x, c2_y}; project_pt(pts[8]);

            int rem = n - 9;
            double rot = unif(rng) * 2.0 * 3.141592653589793;
            for (int r = 0; r < rem; ++r) {
                if (mode == 1 && r == rem - 1) {
                    // 1 central point near (0.28, 0.28)
                    pts[9 + r] = {0.25 + 0.12 * unif(rng), 0.25 + 0.12 * unif(rng)};
                } else {
                    int k_ring = (mode == 1) ? (rem - 1) : rem;
                    double ang = rot + r * (2.0 * 3.141592653589793 / std::max(1, k_ring)) + 0.15 * norm(rng);
                    double rad = 0.14 + 0.08 * unif(rng);
                    pts[9 + r] = {0.31 + rad * std::cos(ang), 0.31 + rad * std::sin(ang)};
                }
                project_pt(pts[9 + r]);
            }
        } else {
            int rem = n - 6;
            double rot = unif(rng) * 2.0 * 3.141592653589793;
            for (int r = 0; r < rem; ++r) {
                double ang = rot + r * (2.0 * 3.141592653589793 / rem) + 0.2 * norm(rng);
                double rad = 0.12 + 0.14 * unif(rng);
                Pt p = {0.30 + rad * std::cos(ang), 0.30 + rad * std::sin(ang)};
                project_pt(p);
                pts[6 + r] = p;
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

std::vector<Pt> build_search_grid(int res = 110) {
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

std::vector<Pt> top_pockets_for_subset(const std::vector<Pt>& sub, const std::vector<Pt>& grid, int top_k = 8) {
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
            if (dx * dx + dy * dy < 0.04 * 0.04) { far_enough = false; break; }
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
    int n_iters = 50
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

        int move_type = it % 4;
        bool use_soft = false;
        if (move_type == 0) {
            // 1-vertex relocation from bottleneck triangle to top pocket
            int v_del = b_verts[rng() % 3];
            std::vector<Pt> sub;
            for (int i = 0; i < n; ++i) if (i != v_del) sub.push_back(cand[i]);
            auto pockets = top_pockets_for_subset(sub, grid, 4);
            cand[v_del] = pockets[rng() % pockets.size()];
            use_soft = true;
        } else if (move_type == 1) {
            // 2-vertex Ruin-and-Recreate (remove 2 vertices, insert sequentially into top pockets!)
            int v1 = b_verts[rng() % 3];
            int v2 = rng() % n;
            while (v2 == v1) v2 = rng() % n;
            std::vector<Pt> sub;
            for (int i = 0; i < n; ++i) if (i != v1 && i != v2) sub.push_back(cand[i]);
            auto p1_list = top_pockets_for_subset(sub, grid, 3);
            Pt q1 = p1_list[rng() % p1_list.size()];
            sub.push_back(q1);
            auto p2_list = top_pockets_for_subset(sub, grid, 3);
            Pt q2 = p2_list[rng() % p2_list.size()];
            cand[v1] = q1;
            cand[v2] = q2;
            use_soft = true;
        } else if (move_type == 2) {
            // Perturb 2 or 3 vertices
            double sigma = (it < n_iters / 2) ? 0.030 : 0.010;
            for (int k = 0; k < 2; ++k) {
                int v = b_verts[(it + k) % 3];
                cand[v].x += sigma * norm(rng);
                cand[v].y += sigma * norm(rng);
                project_pt(cand[v]);
            }
        } else {
            // Global random vertex relocation + soft relaxation
            int v_del = rng() % n;
            std::vector<Pt> sub;
            for (int i = 0; i < n; ++i) if (i != v_del) sub.push_back(cand[i]);
            auto pockets = top_pockets_for_subset(sub, grid, 5);
            Pt q = pockets[rng() % pockets.size()];
            q.x += 0.01 * norm(rng);
            q.y += 0.01 * norm(rng);
            project_pt(q);
            cand[v_del] = q;
            use_soft = true;
        }

        local_polish(cand, tris, 1000, use_soft);
        double val = eval_min_area(cand, tris);
        if (val > best_val) {
            best_val = val;
            best_pts = cand;
        }
    }
    pts = best_pts;
}

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();
    const auto grid = build_search_grid(96);

    const auto tris13 = make_triplets(13);
    const auto tris14 = make_triplets(14);
    const auto tris15 = make_triplets(15);

    std::vector<std::pair<double, std::vector<Pt>>> pool13(160);
    std::vector<std::pair<double, std::vector<Pt>>> pool14(160);
    std::vector<std::pair<double, std::vector<Pt>>> pool15(120);

    // 1. Initial multi-start on n=13, n=14, n=15
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < 160; ++r) {
        std::mt19937_64 rng(10000 + r * 97);
        auto pts = seed_topological(13, rng);
        local_polish(pts, tris13, 1000, true);
        ruin_and_recreate_chain(pts, tris13, grid, rng, 18);
        pool13[r] = {eval_min_area(pts, tris13), pts};
    }

    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < 160; ++r) {
        std::mt19937_64 rng(30000 + r * 113);
        auto pts = seed_topological(14, rng);
        local_polish(pts, tris14, 1000, true);
        ruin_and_recreate_chain(pts, tris14, grid, rng, 20);
        pool14[r] = {eval_min_area(pts, tris14), pts};
    }

    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < 120; ++r) {
        std::mt19937_64 rng(50000 + r * 127);
        auto pts = seed_topological(15, rng);
        local_polish(pts, tris15, 1000, true);
        ruin_and_recreate_chain(pts, tris15, grid, rng, 16);
        pool15[r] = {eval_min_area(pts, tris15), pts};
    }

    auto sort_pool = [](auto& p) {
        std::sort(p.begin(), p.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    };
    sort_pool(pool13);
    sort_pool(pool14);
    sort_pool(pool15);

    std::cout << "[Round 1 Initial] n=13 best=" << pool13[0].first
              << " | n=14 best=" << pool14[0].first
              << " | n=15 best=" << pool15[0].first << std::endl;

    // 2. Downward Vertical Seeding: 14 -> 13 and 15 -> 14
    std::vector<std::pair<double, std::vector<Pt>>> down13(20 * 14);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < 20 * 14; ++idx) {
        int e_id = idx / 14, v_del = idx % 14;
        std::mt19937_64 rng(70000 + idx * 41);
        const auto& src = pool14[e_id].second;
        std::vector<Pt> sub;
        for (int i = 0; i < 14; ++i) if (i != v_del) sub.push_back(src[i]);
        local_polish(sub, tris13, 1000, false);
        ruin_and_recreate_chain(sub, tris13, grid, rng, 22);
        down13[idx] = {eval_min_area(sub, tris13), sub};
    }
    for (const auto& x : down13) pool13.push_back(x);
    sort_pool(pool13);

    std::vector<std::pair<double, std::vector<Pt>>> down14(20 * 15);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < 20 * 15; ++idx) {
        int e_id = idx / 15, v_del = idx % 15;
        std::mt19937_64 rng(90000 + idx * 53);
        const auto& src = pool15[e_id].second;
        std::vector<Pt> sub;
        for (int i = 0; i < 15; ++i) if (i != v_del) sub.push_back(src[i]);
        local_polish(sub, tris14, 1100, false);
        ruin_and_recreate_chain(sub, tris14, grid, rng, 26);
        down14[idx] = {eval_min_area(sub, tris14), sub};
    }
    for (const auto& x : down14) pool14.push_back(x);
    sort_pool(pool14);

    std::cout << "[Round 2 Downward Vertical (14->13, 15->14)] n=13 best=" << pool13[0].first
              << " | n=14 best=" << pool14[0].first << std::endl;

    // 3. Upward Vertical Seeding (13 -> 14 via Pocket Insertion + Vertex Mitosis from NEW n=13 elites!)
    std::vector<std::pair<double, std::vector<Pt>>> up14(24 * 12);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < 24 * 12; ++idx) {
        int e_id = idx / 12;
        int mode = idx % 12;
        std::mt19937_64 rng(120000 + idx * 67);
        const auto& base13 = pool13[e_id].second;
        std::vector<Pt> cand14 = base13;

        if (mode < 6) {
            // Top-6 distinct pockets of the n=13 elite
            auto pockets = top_pockets_for_subset(base13, grid, 6);
            cand14.push_back(pockets[mode % pockets.size()]);
        } else {
            // Vertex Mitosis: split an interior vertex of base13 into 2 points!
            std::vector<int> inner_idx;
            for (int i = 0; i < 13; ++i) {
                if (base13[i].x > 0.02 && base13[i].y > 0.02 && (base13[i].x + base13[i].y) < 0.98)
                    inner_idx.push_back(i);
            }
            int v_split = inner_idx.empty() ? (idx % 13) : inner_idx[mode % inner_idx.size()];
            double ang = (mode - 6) * (3.141592653589793 / 6.0);
            double dx = 0.045 * std::cos(ang), dy = 0.045 * std::sin(ang);
            Pt p_a = {base13[v_split].x + dx, base13[v_split].y + dy};
            Pt p_b = {base13[v_split].x - dx, base13[v_split].y - dy};
            project_pt(p_a);
            project_pt(p_b);
            cand14[v_split] = p_a;
            cand14.push_back(p_b);
        }
        local_polish(cand14, tris14, 1200, true);
        ruin_and_recreate_chain(cand14, tris14, grid, rng, 35);
        up14[idx] = {eval_min_area(cand14, tris14), cand14};
    }
    for (const auto& x : up14) pool14.push_back(x);
    sort_pool(pool14);

    std::cout << "[Round 3 Upward Vertical + Mitosis (13->14)] n=14 Top-5: ";
    for (int i = 0; i < 5; ++i) std::cout << std::fixed << std::setprecision(8) << pool14[i].first << " ";
    std::cout << std::endl;

    // 4. Second Downward Pass (14 -> 13 from any newly improved n=14 elites) + Final Deep R&R on n=13 & n=14
    std::vector<std::pair<double, std::vector<Pt>>> down13_b(16 * 14);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < 16 * 14; ++idx) {
        int e_id = idx / 14, v_del = idx % 14;
        std::mt19937_64 rng(160000 + idx * 79);
        const auto& src = pool14[e_id].second;
        std::vector<Pt> sub;
        for (int i = 0; i < 14; ++i) if (i != v_del) sub.push_back(src[i]);
        local_polish(sub, tris13, 1000, false);
        ruin_and_recreate_chain(sub, tris13, grid, rng, 30);
        down13_b[idx] = {eval_min_area(sub, tris13), sub};
    }
    for (const auto& x : down13_b) pool13.push_back(x);
    sort_pool(pool13);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 48; ++i) {
        std::mt19937_64 rng(200000 + i * 97);
        auto pts = pool14[i % 16].second;
        ruin_and_recreate_chain(pts, tris14, grid, rng, 60);
        double v = eval_min_area(pts, tris14);
        #pragma omp critical
        {
            pool14.push_back({v, pts});
        }
    }
    sort_pool(pool14);

    std::cout << "[FINAL C++ Top-5 n=13]: ";
    for (int i = 0; i < 5; ++i) std::cout << std::fixed << std::setprecision(8) << pool13[i].first << " ";
    std::cout << "\n[FINAL C++ Top-5 n=14]: ";
    for (int i = 0; i < 5; ++i) std::cout << std::fixed << std::setprecision(8) << pool14[i].first << " ";
    std::cout << std::endl;

    std::ofstream out("heilbronn_cpp_candidates.json");
    out << std::setprecision(17) << "{\n  \"n13\": [\n";
    for (int i = 0; i < 25; ++i) {
        out << "    [";
        for (int p = 0; p < 13; ++p) {
            out << "[" << pool13[i].second[p].x << "," << pool13[i].second[p].y << "]" << (p < 12 ? "," : "");
        }
        out << "]" << (i < 24 ? ",\n" : "\n");
    }
    out << "  ],\n  \"n14\": [\n";
    for (int i = 0; i < 25; ++i) {
        out << "    [";
        for (int p = 0; p < 14; ++p) {
            out << "[" << pool14[i].second[p].x << "," << pool14[i].second[p].y << "]" << (p < 13 ? "," : "");
        }
        out << "]" << (i < 24 ? ",\n" : "\n");
    }
    out << "  ]\n}\n";
    out.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "C++ Search completed in " << std::fixed << std::setprecision(2) << elapsed << "s\n";
    return 0;
}
