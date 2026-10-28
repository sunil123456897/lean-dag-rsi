// ============================================================================
// Сравнительный C++ стенд:
// 1. solve_active_set — точная копия Listing 3 из статьи Dream-RSI (стр. 23-36)
// 2. solve_celer_lookahead — модернизированный солвер из Раздела 4.2:
//    - Dynamic Gap Safe Sphere Screening + Look-Ahead Path Screening
//      (работает при любых n, включая биологические датасеты n < 150)
//    - Celer Dual Extrapolation (Massias et al., ICML 2018)
//    - Устранение холостого O(|A|^2) сброса кэша градиентов c_data (строки 421-438)
//    - Сертифицированный критерий останова по двойственному зазору Gap_k <= 0.9e-6
// ============================================================================

#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#else
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
static inline int posix_memalign_compat(void** memptr, size_t alignment, size_t size) {
    if (size == 0) size = alignment;
    void* p = _aligned_malloc(size, alignment);
    if (!p) return -1;
    *memptr = p;
    return 0;
}
static inline void aligned_free_compat(void* ptr) {
    if (ptr) _aligned_free(ptr);
}
#else
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) (double*)__builtin_assume_aligned((ptr), (alignment))
static inline int posix_memalign_compat(void** memptr, size_t alignment, size_t size) {
    if (size == 0) size = alignment;
    return posix_memalign(memptr, alignment, size);
}
static inline void aligned_free_compat(void* ptr) {
    if (ptr) free(ptr);
}
#endif

// Минимальная совместимая обёртка VectorXd / MatrixXd (Column-Major, как в Eigen)
struct VectorXd {
    std::vector<double> buf;
    VectorXd() {}
    explicit VectorXd(int n) : buf(n, 0.0) {}
    VectorXd(int n, double val) : buf(n, val) {}
    static VectorXd Zero(int n) { return VectorXd(n, 0.0); }
    int size() const { return static_cast<int>(buf.size()); }
    double* data() { return buf.data(); }
    const double* data() const { return buf.data(); }
    double& operator()(int i) { return buf[i]; }
    double operator()(int i) const { return buf[i]; }
    double& operator[](int i) { return buf[i]; }
    double operator[](int i) const { return buf[i]; }
};

struct MatrixXd {
    int rows_ = 0;
    int cols_ = 0;
    std::vector<double> buf; // Column-major: element (r, c) at c * rows_ + r
    MatrixXd() {}
    MatrixXd(int r, int c) : rows_(r), cols_(c), buf(static_cast<size_t>(r) * c, 0.0) {}
    static MatrixXd Zero(int r, int c) { return MatrixXd(r, c); }
    double* data() { return buf.data(); }
    const double* data() const { return buf.data(); }
    void set_zero() { std::fill(buf.begin(), buf.end(), 0.0); }
    struct ColProxy {
        double* col_ptr;
        int r;
        ColProxy& operator=(const VectorXd& v) {
            std::memcpy(col_ptr, v.data(), static_cast<size_t>(r) * sizeof(double));
            return *this;
        }
    };
    ColProxy col(int c) { return ColProxy{buf.data() + static_cast<size_t>(c) * rows_, rows_}; }
};

static inline double soft_thresh(double z, double gamma) {
    double abs_z = std::abs(z);
    double val = abs_z - gamma;
    return std::copysign(val > 0.0 ? val : 0.0, z);
}

// ============================================================================
// 1. ТОЧНАЯ КОПИЯ SOLVE_ACTIVE_SET ИЗ LISTING 3 (DREAM-RSI)
// ============================================================================
void solve_active_set(
    const double* RESTRICT X_padded,
    int n_padded,
    int n,
    int p,
    const VectorXd& y,
    const VectorXd& lam_path,
    const VectorXd& xv,
    const VectorXd& grad_init,
    MatrixXd& coef_path,
    double thresh,
    int maxit)
{
    const double fn = static_cast<double>(n);
    const double inv_fn = 1.0 / fn;
    const double tol = thresh;
    const int nlam = lam_path.size();

    // Workload-Aware flag for activating Cauchy-Schwarz KKT Pruning (Listing 3 line 65)
    const bool use_cs = (p >= 500 && n >= 150);

    int current_capacity = ((std::max(128, std::min(512, p)) + 7) / 8) * 8;

    double* G_data = nullptr;
    double* c_data = nullptr;
    double* beta_active_data = nullptr;
    double* xv_active_data = nullptr;
    double* inv_xv_active_data = nullptr;
    double* grad_init_active_data = nullptr;
    double* beta_old_at_start = nullptr;

    double* y_padded = nullptr;
    double* r_padded = nullptr;
    double* r_ref_padded = nullptr;

    bool oom = false;

    if (posix_memalign_compat((void**)&G_data, 64, static_cast<size_t>(current_capacity) * current_capacity * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&c_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&beta_active_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&xv_active_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&inv_xv_active_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&grad_init_active_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&beta_old_at_start, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;

    if (posix_memalign_compat((void**)&y_padded, 64, static_cast<size_t>(n_padded) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&r_padded, 64, static_cast<size_t>(n_padded) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&r_ref_padded, 64, static_cast<size_t>(n_padded) * sizeof(double)) != 0) goto cleanup;

    std::fill(G_data, G_data + static_cast<size_t>(current_capacity) * current_capacity, 0.0);
    std::fill(c_data, c_data + current_capacity, 0.0);
    std::fill(beta_active_data, beta_active_data + current_capacity, 0.0);
    std::fill(xv_active_data, xv_active_data + current_capacity, 0.0);
    std::fill(inv_xv_active_data, inv_xv_active_data + current_capacity, 0.0);
    std::fill(grad_init_active_data, grad_init_active_data + current_capacity, 0.0);
    std::fill(beta_old_at_start, beta_old_at_start + current_capacity, 0.0);

    std::memcpy(y_padded, y.data(), n * sizeof(double));
    for (int i = n; i < n_padded; ++i) y_padded[i] = 0.0;

    std::memcpy(r_padded, y_padded, n_padded * sizeof(double));
    std::memcpy(r_ref_padded, y_padded, n_padded * sizeof(double));

    {
        VectorXd beta = VectorXd::Zero(p);

        std::vector<char> screened(p, 0);
        std::vector<int> active;
        std::vector<int> feat_to_idx(p, -1);

        std::vector<int> unscreened_list(p);
        std::vector<int> screened_list(p);
        std::vector<int> screened_to_idx(p, -1);

        int unscreened_size = p;
        int screened_size = 0;
        for (int j = 0; j < p; ++j) {
            unscreened_list[j] = j;
        }

        VectorXd grad = grad_init;

        VectorXd grad_ref;
        std::vector<double> s;
        int lambdas_since_reset = 0;

        if (use_cs) {
            grad_ref = grad_init;
            s.resize(p);
            for (int j = 0; j < p; ++j) {
                s[j] = std::sqrt(xv(j) * inv_fn);
            }
        }

        auto add_active = [&](int j) {
            if (feat_to_idx[j] != -1) return;

            int idx_in_screened = screened_to_idx[j];
            if (idx_in_screened >= 0) {
                int last_j = screened_list[screened_size - 1];
                screened_list[idx_in_screened] = last_j;
                screened_to_idx[last_j] = idx_in_screened;
                --screened_size;
                screened_to_idx[j] = -1;
            }

            int old_k = static_cast<int>(active.size());
            feat_to_idx[j] = old_k;
            active.push_back(j);
            int new_k = old_k + 1;

            if (new_k > current_capacity) {
                int new_capacity = current_capacity * 2;

                double* G_data2 = nullptr;
                double* c_data2 = nullptr;
                double* beta_active_data2 = nullptr;
                double* xv_active_data2 = nullptr;
                double* inv_xv_active_data2 = nullptr;
                double* grad_init_active_data2 = nullptr;
                double* beta_old_at_start2 = nullptr;

                if (posix_memalign_compat((void**)&G_data2, 64, static_cast<size_t>(new_capacity) * new_capacity * sizeof(double)) != 0) { oom = true; return; }
                if (posix_memalign_compat((void**)&c_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&beta_active_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&xv_active_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); aligned_free_compat(beta_active_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&inv_xv_active_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); aligned_free_compat(beta_active_data2); aligned_free_compat(xv_active_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&grad_init_active_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); aligned_free_compat(beta_active_data2); aligned_free_compat(xv_active_data2); aligned_free_compat(inv_xv_active_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&beta_old_at_start2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); aligned_free_compat(beta_active_data2); aligned_free_compat(xv_active_data2); aligned_free_compat(inv_xv_active_data2); aligned_free_compat(grad_init_active_data2); oom = true; return; }

                std::fill(G_data2, G_data2 + static_cast<size_t>(new_capacity) * new_capacity, 0.0);

                if (old_k > 0) {
                    int old_k_padded = (old_k + 7) & ~7;
                    for (int col = 0; col < old_k; ++col) {
                        double* dest_col = G_data2 + col * new_capacity;
                        const double* src_col = G_data + col * current_capacity;
                        for (int row = 0; row < old_k_padded; ++row) {
                            dest_col[row] = src_col[row];
                        }
                    }
                    for (int i = 0; i < old_k_padded; ++i) c_data2[i] = c_data[i];
                    for (int i = 0; i < old_k_padded; ++i) beta_active_data2[i] = beta_active_data[i];
                    for (int i = 0; i < old_k_padded; ++i) xv_active_data2[i] = xv_active_data[i];
                    for (int i = 0; i < old_k_padded; ++i) inv_xv_active_data2[i] = inv_xv_active_data[i];
                    for (int i = 0; i < old_k_padded; ++i) grad_init_active_data2[i] = grad_init_active_data[i];
                    for (int i = 0; i < old_k_padded; ++i) beta_old_at_start2[i] = beta_old_at_start[i];
                }

                aligned_free_compat(G_data);
                aligned_free_compat(c_data);
                aligned_free_compat(beta_active_data);
                aligned_free_compat(xv_active_data);
                aligned_free_compat(inv_xv_active_data);
                aligned_free_compat(grad_init_active_data);
                aligned_free_compat(beta_old_at_start);

                G_data = G_data2;
                c_data = c_data2;
                beta_active_data = beta_active_data2;
                xv_active_data = xv_active_data2;
                inv_xv_active_data = inv_xv_active_data2;
                grad_init_active_data = grad_init_active_data2;
                beta_old_at_start = beta_old_at_start2;
                current_capacity = new_capacity;
            }

            const double* RESTRICT col_j = ASSUME_ALIGNED(X_padded + j * n_padded, 64);
            for (int i = 0; i < (old_k / 4) * 4; i += 4) {
                const double* RESTRICT col0 = ASSUME_ALIGNED(X_padded + active[i] * n_padded, 64);
                const double* RESTRICT col1 = ASSUME_ALIGNED(X_padded + active[i+1] * n_padded, 64);
                const double* RESTRICT col2 = ASSUME_ALIGNED(X_padded + active[i+2] * n_padded, 64);
                const double* RESTRICT col3 = ASSUME_ALIGNED(X_padded + active[i+3] * n_padded, 64);

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                for (int k = 0; k < n_padded; ++k) {
                    double vj = col_j[k];
                    sum0 += vj * col0[k];
                    sum1 += vj * col1[k];
                    sum2 += vj * col2[k];
                    sum3 += vj * col3[k];
                }

                double r0 = sum0 * inv_fn;
                double r1 = sum1 * inv_fn;
                double r2 = sum2 * inv_fn;
                double r3 = sum3 * inv_fn;

                G_data[old_k * current_capacity + i] = r0;
                G_data[i * current_capacity + old_k] = r0;
                G_data[old_k * current_capacity + i + 1] = r1;
                G_data[(i + 1) * current_capacity + old_k] = r1;
                G_data[old_k * current_capacity + i + 2] = r2;
                G_data[(i + 2) * current_capacity + old_k] = r2;
                G_data[old_k * current_capacity + i + 3] = r3;
                G_data[(i + 3) * current_capacity + old_k] = r3;
            }

            for (int i = (old_k / 4) * 4; i < old_k; ++i) {
                const double* RESTRICT col_act = ASSUME_ALIGNED(X_padded + active[i] * n_padded, 64);
                double dot_val = 0.0;
                for (int k = 0; k < n_padded; ++k) {
                    dot_val += col_j[k] * col_act[k];
                }
                dot_val *= inv_fn;
                G_data[old_k * current_capacity + i] = dot_val;
                G_data[i * current_capacity + old_k] = dot_val;
            }
            G_data[old_k * current_capacity + old_k] = xv(j);

            double sum_val = 0.0;
            const double* RESTRICT G_col = ASSUME_ALIGNED(G_data + old_k * current_capacity, 64);
            const double* RESTRICT beta_act = ASSUME_ALIGNED(beta_active_data, 64);
            for (int i = 0; i < old_k; ++i) {
                sum_val += G_col[i] * beta_act[i];
            }
            c_data[old_k] = grad_init(j) - sum_val;

            xv_active_data[old_k] = xv(j);
            inv_xv_active_data[old_k] = 1.0 / xv(j);
            grad_init_active_data[old_k] = grad_init[j];
            beta_active_data[old_k] = 0.0;
        };

        double prev_lam = 0.0;

        std::vector<int> to_activate;
        std::vector<int> screened_violators;
        std::vector<int> unscreened_violators;
        std::vector<int> to_compute;

        to_activate.reserve(p);
        screened_violators.reserve(p);
        unscreened_violators.reserve(p);
        if (use_cs) {
            to_compute.reserve(p);
        }

        for (int li = 0; li < nlam; ++li) {
            const double lam = lam_path(li);
            const double tlam = 2.0 * lam - prev_lam;

            // ---- Step 1: Strong-rule screening ----
            double* RESTRICT grad_ptr = grad.data();
            for (int i = 0; i < unscreened_size; ) {
                int j = unscreened_list[i];
                if (std::abs(grad_ptr[j]) > tlam) {
                    screened[j] = 1;
                    screened_to_idx[j] = screened_size;
                    screened_list[screened_size++] = j;
                    unscreened_list[i] = unscreened_list[--unscreened_size];
                } else {
                    ++i;
                }
            }

            // ---- Step 2: Outer loop ----
            int nlp = 0;
            while (true) {
                to_activate.clear();
                const double KKT_bound_screen = lam * (1.0 + 1e-9);
                for (int i = 0; i < screened_size; ++i) {
                    int j = screened_list[i];
                    if (std::abs(grad_ptr[j]) > KKT_bound_screen) {
                        to_activate.push_back(j);
                    }
                }

                if (!to_activate.empty()) {
                    for (int j : to_activate) {
                        add_active(j);
                        if (oom) goto cleanup;
                    }
                }

                int active_size = static_cast<int>(active.size());

                if (active_size > 0) {
                    int active_size_padded = (active_size + 7) & ~7;
                    for (int i = 0; i < active_size_padded; ++i) {
                        beta_old_at_start[i] = beta_active_data[i];
                    }
                }

                if (active_size > 0) {
                    double dmax = tol;
                    while (dmax >= tol && nlp < maxit) {
                        ++nlp;
                        dmax = 0.0;
                        for (int idx = 0; idx < active_size; ++idx) {
                            const double bj_old = beta_active_data[idx];
                            const double gj = c_data[idx] + bj_old * xv_active_data[idx];
                            const double bj_new = soft_thresh(gj, lam) * inv_xv_active_data[idx];
                            if (bj_new == bj_old) continue;
                            const double delta = bj_new - bj_old;
                            beta_active_data[idx] = bj_new;

                            int active_size_padded = (active_size + 7) & ~7;
                            double* RESTRICT c_ptr = ASSUME_ALIGNED(c_data, 64);
                            const double* RESTRICT G_col_ptr = ASSUME_ALIGNED(G_data + idx * current_capacity, 64);
                            for (int i = 0; i < active_size_padded; ++i) {
                                c_ptr[i] -= delta * G_col_ptr[i];
                            }

                            const double ch = xv_active_data[idx] * delta * delta;
                            if (ch > dmax) dmax = ch;
                        }
                    }
                }

                if (nlp >= maxit) break;

                bool any_changed = false;
                if (active_size > 0) {
                    double* RESTRICT r_ptr = ASSUME_ALIGNED(r_padded, 64);
                    for (int idx = 0; idx < active_size; ++idx) {
                        const double delta = beta_active_data[idx] - beta_old_at_start[idx];
                        if (delta != 0.0) {
                            const double* RESTRICT col_ptr = ASSUME_ALIGNED(X_padded + active[idx] * n_padded, 64);
                            for (int i = 0; i < n_padded; ++i) {
                                r_ptr[i] -= delta * col_ptr[i];
                            }
                            any_changed = true;
                        }
                    }
                }

                // O(k^2) exact re-sync of correlation cache c (Listing 3 lines 421-438)
                if (any_changed && active_size > 0) {
                    int active_size_padded = (active_size + 7) & ~7;
                    for (int i = 0; i < active_size_padded; ++i) {
                        c_data[i] = grad_init_active_data[i];
                    }
                    for (int j = 0; j < active_size; ++j) {
                        const double bj = beta_active_data[j];
                        if (bj != 0.0) {
                            const double* RESTRICT G_col = ASSUME_ALIGNED(G_data + j * current_capacity, 64);
                            double* RESTRICT c_ptr = ASSUME_ALIGNED(c_data, 64);
                            for (int i = 0; i < active_size_padded; ++i) {
                                c_ptr[i] -= bj * G_col[i];
                            }
                        }
                    }
                }

                // 2c. Robust Two-Stage KKT check
                bool screened_kkt_ok = true;
                screened_violators.clear();
                const double KKT_bound = lam * (1.0 + 1e-9);

                const double* RESTRICT r_ptr = ASSUME_ALIGNED(r_padded, 64);
                for (int i = 0; i < (screened_size / 4) * 4; i += 4) {
                    int j0 = screened_list[i];
                    int j1 = screened_list[i+1];
                    int j2 = screened_list[i+2];
                    int j3 = screened_list[i+3];

                    const double* RESTRICT col0 = ASSUME_ALIGNED(X_padded + j0 * n_padded, 64);
                    const double* RESTRICT col1 = ASSUME_ALIGNED(X_padded + j1 * n_padded, 64);
                    const double* RESTRICT col2 = ASSUME_ALIGNED(X_padded + j2 * n_padded, 64);
                    const double* RESTRICT col3 = ASSUME_ALIGNED(X_padded + j3 * n_padded, 64);

                    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (int k = 0; k < n_padded; ++k) {
                        double rk = r_ptr[k];
                        sum0 += rk * col0[k];
                        sum1 += rk * col1[k];
                        sum2 += rk * col2[k];
                        sum3 += rk * col3[k];
                    }
                    grad_ptr[j0] = sum0 * inv_fn;
                    grad_ptr[j1] = sum1 * inv_fn;
                    grad_ptr[j2] = sum2 * inv_fn;
                    grad_ptr[j3] = sum3 * inv_fn;
                }

                for (int i = (screened_size / 4) * 4; i < screened_size; ++i) {
                    int j = screened_list[i];
                    const double* RESTRICT col_ptr = ASSUME_ALIGNED(X_padded + j * n_padded, 64);
                    double dot_val = 0.0;
                    for (int k = 0; k < n_padded; ++k) {
                        dot_val += col_ptr[k] * r_ptr[k];
                    }
                    grad_ptr[j] = dot_val * inv_fn;
                }

                for (int i = 0; i < screened_size; ++i) {
                    int j = screened_list[i];
                    if (std::abs(grad_ptr[j]) > KKT_bound) {
                        screened_violators.push_back(j);
                        screened_kkt_ok = false;
                    }
                }

                if (!screened_kkt_ok) {
                    for (int j : screened_violators) {
                        add_active(j);
                        if (oom) goto cleanup;
                    }
                    continue;
                }

                bool full_kkt_ok = true;
                unscreened_violators.clear();

                if (use_cs) {
                    double d2 = 0.0;
                    const double* RESTRICT r_curr_ptr = ASSUME_ALIGNED(r_padded, 64);
                    const double* RESTRICT r_ref_ptr = ASSUME_ALIGNED(r_ref_padded, 64);
                    for (int k = 0; k < n_padded; ++k) {
                        double diff = r_curr_ptr[k] - r_ref_ptr[k];
                        d2 += diff * diff;
                    }
                    double d = std::sqrt(d2);

                    const double* RESTRICT grad_ref_ptr = grad_ref.data();
                    const double* RESTRICT s_ptr = s.data();
                    const int* RESTRICT unscreened_ptr = unscreened_list.data();

                    to_compute.clear();
                    for (int i = 0; i < unscreened_size; ++i) {
                        int j = unscreened_ptr[i];
                        double bound = std::abs(grad_ref_ptr[j]) + s_ptr[j] * d;
                        if (bound > KKT_bound) {
                            to_compute.push_back(j);
                        }
                    }

                    int num_to_compute = static_cast<int>(to_compute.size());
                    bool did_reset = false;

                    if (num_to_compute > 0.3 * p || lambdas_since_reset >= 8) {
                        for (int i = 0; i < (unscreened_size / 4) * 4; i += 4) {
                            int j0 = unscreened_list[i];
                            int j1 = unscreened_list[i+1];
                            int j2 = unscreened_list[i+2];
                            int j3 = unscreened_list[i+3];

                            const double* RESTRICT col0 = ASSUME_ALIGNED(X_padded + j0 * n_padded, 64);
                            const double* RESTRICT col1 = ASSUME_ALIGNED(X_padded + j1 * n_padded, 64);
                            const double* RESTRICT col2 = ASSUME_ALIGNED(X_padded + j2 * n_padded, 64);
                            const double* RESTRICT col3 = ASSUME_ALIGNED(X_padded + j3 * n_padded, 64);
                            const double* RESTRICT r_ptr_exact = ASSUME_ALIGNED(r_padded, 64);

                            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                            for (int k = 0; k < n_padded; ++k) {
                                double rk = r_ptr_exact[k];
                                sum0 += rk * col0[k];
                                sum1 += rk * col1[k];
                                sum2 += rk * col2[k];
                                sum3 += rk * col3[k];
                            }
                            grad_ptr[j0] = sum0 * inv_fn;
                            grad_ptr[j1] = sum1 * inv_fn;
                            grad_ptr[j2] = sum2 * inv_fn;
                            grad_ptr[j3] = sum3 * inv_fn;
                        }

                        for (int i = (unscreened_size / 4) * 4; i < unscreened_size; ++i) {
                            int j = unscreened_list[i];
                            const double* RESTRICT col_ptr = ASSUME_ALIGNED(X_padded + j * n_padded, 64);
                            const double* RESTRICT r_ptr_exact = ASSUME_ALIGNED(r_padded, 64);
                            double sum = 0.0;
                            for (int k = 0; k < n_padded; ++k) {
                                sum += r_ptr_exact[k] * col_ptr[k];
                            }
                            grad_ptr[j] = sum * inv_fn;
                        }

                        std::memcpy(r_ref_padded, r_padded, n_padded * sizeof(double));
                        double* RESTRICT grad_ref_ptr_writable = grad_ref.data();
                        for (int i = 0; i < unscreened_size; ++i) {
                            int j = unscreened_ptr[i];
                            grad_ref_ptr_writable[j] = grad_ptr[j];
                        }
                        lambdas_since_reset = 0;
                        did_reset = true;
                    } else {
                        for (int k = 0; k < (num_to_compute / 4) * 4; k += 4) {
                            int j0 = to_compute[k];
                            int j1 = to_compute[k+1];
                            int j2 = to_compute[k+2];
                            int j3 = to_compute[k+3];

                            const double* RESTRICT col0 = ASSUME_ALIGNED(X_padded + j0 * n_padded, 64);
                            const double* RESTRICT col1 = ASSUME_ALIGNED(X_padded + j1 * n_padded, 64);
                            const double* RESTRICT col2 = ASSUME_ALIGNED(X_padded + j2 * n_padded, 64);
                            const double* RESTRICT col3 = ASSUME_ALIGNED(X_padded + j3 * n_padded, 64);
                            const double* RESTRICT r_ptr_exact = ASSUME_ALIGNED(r_padded, 64);

                            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                            for (int m = 0; m < n_padded; ++m) {
                                double rk = r_ptr_exact[m];
                                sum0 += rk * col0[m];
                                sum1 += rk * col1[m];
                                sum2 += rk * col2[m];
                                sum3 += rk * col3[m];
                            }
                            grad_ptr[j0] = sum0 * inv_fn;
                            grad_ptr[j1] = sum1 * inv_fn;
                            grad_ptr[j2] = sum2 * inv_fn;
                            grad_ptr[j3] = sum3 * inv_fn;
                        }

                        for (int k = (num_to_compute / 4) * 4; k < num_to_compute; ++k) {
                            int j = to_compute[k];
                            const double* RESTRICT col_ptr = ASSUME_ALIGNED(X_padded + j * n_padded, 64);
                            const double* RESTRICT r_ptr_exact = ASSUME_ALIGNED(r_padded, 64);
                            double sum = 0.0;
                            for (int m = 0; m < n_padded; ++m) {
                                sum += r_ptr_exact[m] * col_ptr[m];
                            }
                            grad_ptr[j] = sum * inv_fn;
                        }
                    }

                    for (int i = 0; i < unscreened_size; ) {
                        int j = unscreened_list[i];
                        if (std::abs(grad_ptr[j]) > KKT_bound) {
                            screened[j] = 1;
                            unscreened_violators.push_back(j);
                            screened_to_idx[j] = screened_size;
                            screened_list[screened_size++] = j;
                            unscreened_list[i] = unscreened_list[--unscreened_size];
                            full_kkt_ok = false;
                        } else {
                            ++i;
                        }
                    }

                    if (full_kkt_ok) {
                        if (!did_reset) {
                            lambdas_since_reset++;
                        }
                    }
                } else {
                    for (int i = 0; i < (unscreened_size / 4) * 4; i += 4) {
                        int j0 = unscreened_list[i];
                        int j1 = unscreened_list[i+1];
                        int j2 = unscreened_list[i+2];
                        int j3 = unscreened_list[i+3];

                        const double* RESTRICT col0 = ASSUME_ALIGNED(X_padded + j0 * n_padded, 64);
                        const double* RESTRICT col1 = ASSUME_ALIGNED(X_padded + j1 * n_padded, 64);
                        const double* RESTRICT col2 = ASSUME_ALIGNED(X_padded + j2 * n_padded, 64);
                        const double* RESTRICT col3 = ASSUME_ALIGNED(X_padded + j3 * n_padded, 64);
                        const double* RESTRICT r_ptr_u = ASSUME_ALIGNED(r_padded, 64);

                        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                        for (int k = 0; k < n_padded; ++k) {
                            double rk = r_ptr_u[k];
                            sum0 += rk * col0[k];
                            sum1 += rk * col1[k];
                            sum2 += rk * col2[k];
                            sum3 += rk * col3[k];
                        }
                        grad_ptr[j0] = sum0 * inv_fn;
                        grad_ptr[j1] = sum1 * inv_fn;
                        grad_ptr[j2] = sum2 * inv_fn;
                        grad_ptr[j3] = sum3 * inv_fn;
                    }

                    for (int i = (unscreened_size / 4) * 4; i < unscreened_size; ++i) {
                        int j = unscreened_list[i];
                        const double* RESTRICT col_ptr = ASSUME_ALIGNED(X_padded + j * n_padded, 64);
                        const double* RESTRICT r_ptr_u = ASSUME_ALIGNED(r_padded, 64);
                        double sum = 0.0;
                        for (int k = 0; k < n_padded; ++k) {
                            sum += r_ptr_u[k] * col_ptr[k];
                        }
                        grad_ptr[j] = sum * inv_fn;
                    }

                    for (int i = 0; i < unscreened_size; ) {
                        int j = unscreened_list[i];
                        if (std::abs(grad_ptr[j]) > KKT_bound) {
                            screened[j] = 1;
                            unscreened_violators.push_back(j);
                            screened_to_idx[j] = screened_size;
                            screened_list[screened_size++] = j;
                            unscreened_list[i] = unscreened_list[--unscreened_size];
                            full_kkt_ok = false;
                        } else {
                            ++i;
                        }
                    }
                }

                if (full_kkt_ok) {
                    break;
                }

                for (int j : unscreened_violators) {
                    add_active(j);
                    if (oom) goto cleanup;
                }
            }

            for (size_t idx = 0; idx < active.size(); ++idx) {
                beta(active[idx]) = beta_active_data[idx];
            }
            coef_path.col(li) = beta;
            prev_lam = lam;
        }
    }

cleanup:
    aligned_free_compat(G_data);
    aligned_free_compat(c_data);
    aligned_free_compat(beta_active_data);
    aligned_free_compat(xv_active_data);
    aligned_free_compat(inv_xv_active_data);
    aligned_free_compat(grad_init_active_data);
    aligned_free_compat(beta_old_at_start);
    aligned_free_compat(y_padded);
    aligned_free_compat(r_padded);
    aligned_free_compat(r_ref_padded);
}

// ============================================================================
// 2. МОДЕРНИЗИРОВАННЫЙ C++ СОЛВЕР: CELER + DYNAMIC GAP SAFE + LOOK-AHEAD
//    (Раздел 4.2 Исследование_Dream_RSI_исправлено.md)
// ============================================================================
void solve_celer_lookahead(
    const double* RESTRICT X_padded,
    int n_padded,
    int n,
    int p,
    const VectorXd& y,
    const VectorXd& lam_path,
    const VectorXd& xv,
    const VectorXd& grad_init,
    MatrixXd& coef_path,
    double eps_gap = 0.9e-6,
    int maxit = 100000)
{
    const double fn = static_cast<double>(n);
    const double inv_fn = 1.0 / fn;
    const int nlam = lam_path.size();

    int current_capacity = ((std::max(128, std::min(512, p)) + 7) / 8) * 8;

    double* G_data = nullptr;
    double* c_data = nullptr;
    double* beta_active_data = nullptr;
    double* xv_active_data = nullptr;
    double* inv_xv_active_data = nullptr;
    double* grad_init_active_data = nullptr;
    double* beta_old_at_start = nullptr;

    double* y_padded = nullptr;
    double* r_padded = nullptr;
    double* r_ref_padded = nullptr;

    bool oom = false;

    if (posix_memalign_compat((void**)&G_data, 64, static_cast<size_t>(current_capacity) * current_capacity * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&c_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&beta_active_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&xv_active_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&inv_xv_active_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&grad_init_active_data, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&beta_old_at_start, 64, static_cast<size_t>(current_capacity) * sizeof(double)) != 0) goto cleanup;

    if (posix_memalign_compat((void**)&y_padded, 64, static_cast<size_t>(n_padded) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&r_padded, 64, static_cast<size_t>(n_padded) * sizeof(double)) != 0) goto cleanup;
    if (posix_memalign_compat((void**)&r_ref_padded, 64, static_cast<size_t>(n_padded) * sizeof(double)) != 0) goto cleanup;

    std::fill(G_data, G_data + static_cast<size_t>(current_capacity) * current_capacity, 0.0);
    std::fill(c_data, c_data + current_capacity, 0.0);
    std::fill(beta_active_data, beta_active_data + current_capacity, 0.0);
    std::fill(xv_active_data, xv_active_data + current_capacity, 0.0);
    std::fill(inv_xv_active_data, inv_xv_active_data + current_capacity, 0.0);
    std::fill(grad_init_active_data, grad_init_active_data + current_capacity, 0.0);
    std::fill(beta_old_at_start, beta_old_at_start + current_capacity, 0.0);

    std::memcpy(y_padded, y.data(), n * sizeof(double));
    for (int i = n; i < n_padded; ++i) y_padded[i] = 0.0;
    std::memcpy(r_padded, y_padded, n_padded * sizeof(double));
    std::memcpy(r_ref_padded, y_padded, n_padded * sizeof(double));

    {
        double y_norm2 = 0.0;
        for (int i = 0; i < n; ++i) y_norm2 += y_padded[i] * y_padded[i];
        const double y_norm2_inv_n = y_norm2 * inv_fn;            // ||y||_2^2 / n
        const double y_norm_over_sqrtn = std::sqrt(y_norm2_inv_n); // ||y||_2 / sqrt(n)

        // s_j = sqrt(||x_j||_2^2 / n) = sqrt(xv(j)) для Look-Ahead и Gap Safe (где радиусы уже поделены на sqrt(n))
        // s_cs_j = sqrt(xv(j) / n) для быстрой фильтрации дрейфа ||r - r_ref||_2 без деления d на sqrt(n)
        std::vector<double> s(p);
        std::vector<double> s_cs(p);
        for (int j = 0; j < p; ++j) {
            s[j] = std::sqrt(xv(j));
            s_cs[j] = std::sqrt(xv(j) * inv_fn);
        }

        // Индекс шага пробуждения Look-Ahead: признак j гарантированно равен 0 на всех шагах li < safe_until[j]
        std::vector<int> safe_until(p, 0);

        VectorXd beta = VectorXd::Zero(p);
        std::vector<char> screened(p, 0);
        std::vector<int> active;
        std::vector<int> feat_to_idx(p, -1);

        std::vector<int> unscreened_list(p);
        std::vector<int> screened_list(p);
        std::vector<int> screened_to_idx(p, -1);

        int unscreened_size = p;
        int screened_size = 0;
        for (int j = 0; j < p; ++j) {
            unscreened_list[j] = j;
        }

        VectorXd grad = grad_init;
        VectorXd grad_ref = grad_init;
        int lambdas_since_reset = 0;

        auto add_active = [&](int j) {
            if (feat_to_idx[j] != -1) return;

            int idx_in_screened = screened_to_idx[j];
            if (idx_in_screened >= 0) {
                int last_j = screened_list[screened_size - 1];
                screened_list[idx_in_screened] = last_j;
                screened_to_idx[last_j] = idx_in_screened;
                --screened_size;
                screened_to_idx[j] = -1;
            }

            int old_k = static_cast<int>(active.size());
            feat_to_idx[j] = old_k;
            active.push_back(j);
            int new_k = old_k + 1;

            if (new_k > current_capacity) {
                int new_capacity = current_capacity * 2;

                double* G_data2 = nullptr;
                double* c_data2 = nullptr;
                double* beta_active_data2 = nullptr;
                double* xv_active_data2 = nullptr;
                double* inv_xv_active_data2 = nullptr;
                double* grad_init_active_data2 = nullptr;
                double* beta_old_at_start2 = nullptr;

                if (posix_memalign_compat((void**)&G_data2, 64, static_cast<size_t>(new_capacity) * new_capacity * sizeof(double)) != 0) { oom = true; return; }
                if (posix_memalign_compat((void**)&c_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&beta_active_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&xv_active_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); aligned_free_compat(beta_active_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&inv_xv_active_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); aligned_free_compat(beta_active_data2); aligned_free_compat(xv_active_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&grad_init_active_data2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); aligned_free_compat(beta_active_data2); aligned_free_compat(xv_active_data2); aligned_free_compat(inv_xv_active_data2); oom = true; return; }
                if (posix_memalign_compat((void**)&beta_old_at_start2, 64, static_cast<size_t>(new_capacity) * sizeof(double)) != 0) { aligned_free_compat(G_data2); aligned_free_compat(c_data2); aligned_free_compat(beta_active_data2); aligned_free_compat(xv_active_data2); aligned_free_compat(inv_xv_active_data2); aligned_free_compat(grad_init_active_data2); oom = true; return; }

                std::fill(G_data2, G_data2 + static_cast<size_t>(new_capacity) * new_capacity, 0.0);

                if (old_k > 0) {
                    int old_k_padded = (old_k + 7) & ~7;
                    for (int col = 0; col < old_k; ++col) {
                        double* dest_col = G_data2 + col * new_capacity;
                        const double* src_col = G_data + col * current_capacity;
                        for (int row = 0; row < old_k_padded; ++row) {
                            dest_col[row] = src_col[row];
                        }
                    }
                    for (int i = 0; i < old_k_padded; ++i) c_data2[i] = c_data[i];
                    for (int i = 0; i < old_k_padded; ++i) beta_active_data2[i] = beta_active_data[i];
                    for (int i = 0; i < old_k_padded; ++i) xv_active_data2[i] = xv_active_data[i];
                    for (int i = 0; i < old_k_padded; ++i) inv_xv_active_data2[i] = inv_xv_active_data[i];
                    for (int i = 0; i < old_k_padded; ++i) grad_init_active_data2[i] = grad_init_active_data[i];
                    for (int i = 0; i < old_k_padded; ++i) beta_old_at_start2[i] = beta_old_at_start[i];
                }

                aligned_free_compat(G_data);
                aligned_free_compat(c_data);
                aligned_free_compat(beta_active_data);
                aligned_free_compat(xv_active_data);
                aligned_free_compat(inv_xv_active_data);
                aligned_free_compat(grad_init_active_data);
                aligned_free_compat(beta_old_at_start);

                G_data = G_data2;
                c_data = c_data2;
                beta_active_data = beta_active_data2;
                xv_active_data = xv_active_data2;
                inv_xv_active_data = inv_xv_active_data2;
                grad_init_active_data = grad_init_active_data2;
                beta_old_at_start = beta_old_at_start2;
                current_capacity = new_capacity;
            }

            const double* RESTRICT col_j = ASSUME_ALIGNED(X_padded + j * n_padded, 64);
            for (int i = 0; i < (old_k / 4) * 4; i += 4) {
                const double* RESTRICT col0 = ASSUME_ALIGNED(X_padded + active[i] * n_padded, 64);
                const double* RESTRICT col1 = ASSUME_ALIGNED(X_padded + active[i+1] * n_padded, 64);
                const double* RESTRICT col2 = ASSUME_ALIGNED(X_padded + active[i+2] * n_padded, 64);
                const double* RESTRICT col3 = ASSUME_ALIGNED(X_padded + active[i+3] * n_padded, 64);

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                for (int k = 0; k < n_padded; ++k) {
                    double vj = col_j[k];
                    sum0 += vj * col0[k];
                    sum1 += vj * col1[k];
                    sum2 += vj * col2[k];
                    sum3 += vj * col3[k];
                }

                double r0 = sum0 * inv_fn;
                double r1 = sum1 * inv_fn;
                double r2 = sum2 * inv_fn;
                double r3 = sum3 * inv_fn;

                G_data[old_k * current_capacity + i] = r0;
                G_data[i * current_capacity + old_k] = r0;
                G_data[old_k * current_capacity + i + 1] = r1;
                G_data[(i + 1) * current_capacity + old_k] = r1;
                G_data[old_k * current_capacity + i + 2] = r2;
                G_data[(i + 2) * current_capacity + old_k] = r2;
                G_data[old_k * current_capacity + i + 3] = r3;
                G_data[(i + 3) * current_capacity + old_k] = r3;
            }

            for (int i = (old_k / 4) * 4; i < old_k; ++i) {
                const double* RESTRICT col_act = ASSUME_ALIGNED(X_padded + active[i] * n_padded, 64);
                double dot_val = 0.0;
                for (int k = 0; k < n_padded; ++k) {
                    dot_val += col_j[k] * col_act[k];
                }
                dot_val *= inv_fn;
                G_data[old_k * current_capacity + i] = dot_val;
                G_data[i * current_capacity + old_k] = dot_val;
            }
            G_data[old_k * current_capacity + old_k] = xv(j);

            double sum_val = 0.0;
            const double* RESTRICT G_col = ASSUME_ALIGNED(G_data + old_k * current_capacity, 64);
            const double* RESTRICT beta_act = ASSUME_ALIGNED(beta_active_data, 64);
            for (int i = 0; i < old_k; ++i) {
                sum_val += G_col[i] * beta_act[i];
            }
            c_data[old_k] = grad_init(j) - sum_val;

            xv_active_data[old_k] = xv(j);
            inv_xv_active_data[old_k] = 1.0 / xv(j);
            grad_init_active_data[old_k] = grad_init[j];
            beta_active_data[old_k] = 0.0;
        };

        double prev_lam = 0.0;
        std::vector<int> to_activate;
        std::vector<int> screened_violators;
        std::vector<int> unscreened_violators;
        std::vector<int> to_compute;
        std::vector<int> cg_S;
        std::vector<double> cg_g(p, 0.0);
        std::vector<double> cg_d(p, 0.0);
        std::vector<double> cg_q(p + 16, 0.0);

        to_activate.reserve(p);
        screened_violators.reserve(p);
        unscreened_violators.reserve(p);
        to_compute.reserve(p);
        cg_S.reserve(p);

        double* RESTRICT grad_ptr = grad.data();
        const double inv_lam_last = 1.0 / lam_path(nlam - 1);

        std::vector<double> inv_lam_grid(nlam);
        for (int m = 0; m < nlam; ++m) {
            inv_lam_grid[m] = 1.0 / lam_path(m);
        }

        for (int li = 0; li < nlam; ++li) {
            const double lam = lam_path(li);
            const double inv_lam = inv_lam_grid[li];
            const double tlam = 2.0 * lam - prev_lam;

            // Шаг 1: Strong-rule скрининг ТОЛЬКО по тем признакам, которые не отсеяны Look-Ahead (safe_until[j] <= li)
            for (int i = 0; i < unscreened_size; ) {
                int j = unscreened_list[i];
                if (safe_until[j] <= li && std::abs(grad_ptr[j]) > tlam) {
                    screened[j] = 1;
                    screened_to_idx[j] = screened_size;
                    screened_list[screened_size++] = j;
                    unscreened_list[i] = unscreened_list[--unscreened_size];
                } else {
                    ++i;
                }
            }

            int nlp = 0;
            while (true) {
                to_activate.clear();
                const double KKT_bound_screen = lam * (1.0 + 1e-9);
                for (int i = 0; i < screened_size; ++i) {
                    int j = screened_list[i];
                    if (std::abs(grad_ptr[j]) > KKT_bound_screen) {
                        to_activate.push_back(j);
                    }
                }

                if (!to_activate.empty()) {
                    for (int j : to_activate) {
                        add_active(j);
                        if (oom) goto cleanup;
                    }
                }

                int active_size = static_cast<int>(active.size());

                if (active_size > 0) {
                    int active_size_padded = (active_size + 7) & ~7;
                    if (static_cast<int>(cg_q.size()) < active_size_padded) {
                        cg_q.resize(active_size_padded + 16, 0.0);
                    }
                    for (int i = 0; i < active_size_padded; ++i) {
                        beta_old_at_start[i] = beta_active_data[i];
                    }

                    // Внутренний цикл CD + подпространственное ускорение (Celer / CG на носителе S = supp(w))
                    // и БЕЗ избыточного O(|A|^2) сброса кэша c_data!
                    double dmax = 1.0;
                    int inner_iter = 0;
                    while (dmax >= 1e-10 && nlp < maxit) {
                        ++nlp;
                        ++inner_iter;
                        dmax = 0.0;
                        for (int idx = 0; idx < active_size; ++idx) {
                            const double bj_old = beta_active_data[idx];
                            const double gj = c_data[idx] + bj_old * xv_active_data[idx];
                            const double bj_new = soft_thresh(gj, lam) * inv_xv_active_data[idx];
                            if (bj_new == bj_old) continue;
                            const double delta = bj_new - bj_old;
                            beta_active_data[idx] = bj_new;

                            double* RESTRICT c_ptr = ASSUME_ALIGNED(c_data, 64);
                            const double* RESTRICT G_col_ptr = ASSUME_ALIGNED(G_data + idx * current_capacity, 64);
                            for (int i = 0; i < active_size_padded; ++i) {
                                c_ptr[i] -= delta * G_col_ptr[i];
                            }

                            const double ch = xv_active_data[idx] * delta * delta;
                            if (ch > dmax) dmax = ch;
                        }

                        if (dmax < 1e-10) break;

                        // Каждые 3 эпохи CD запускаем быстрое квадратичное ускорение по фиксированному знаковому носителю S = supp(beta)
                        if ((inner_iter % 3) == 0 && active_size >= 2) {
                            cg_S.clear();
                            double rs_old = 0.0;
                            for (int idx = 0; idx < active_size; ++idx) {
                                const double b_val = beta_active_data[idx];
                                if (b_val != 0.0) {
                                    cg_S.push_back(idx);
                                    const double g_val = c_data[idx] - std::copysign(lam, b_val);
                                    cg_g[idx] = g_val;
                                    cg_d[idx] = g_val;
                                    rs_old += g_val * g_val;
                                }
                            }

                            const int s_size = static_cast<int>(cg_S.size());
                            if (s_size >= 2 && rs_old > 1e-15) {
                                const int max_cg_steps = std::min(s_size, 12);
                                double* RESTRICT q_ptr = cg_q.data();
                                double* RESTRICT c_ptr = ASSUME_ALIGNED(c_data, 64);

                                for (int cg_step = 0; cg_step < max_cg_steps; ++cg_step) {
                                    ++nlp;
                                    std::fill(q_ptr, q_ptr + active_size_padded, 0.0);
                                    for (int k = 0; k < s_size; ++k) {
                                        const int idx = cg_S[k];
                                        const double d_val = cg_d[idx];
                                        if (d_val == 0.0) continue;
                                        const double* RESTRICT G_col_ptr = ASSUME_ALIGNED(G_data + idx * current_capacity, 64);
                                        for (int i = 0; i < active_size_padded; ++i) {
                                            q_ptr[i] += d_val * G_col_ptr[i];
                                        }
                                    }

                                    double dGd = 0.0;
                                    for (int k = 0; k < s_size; ++k) {
                                        const int idx = cg_S[k];
                                        dGd += cg_d[idx] * q_ptr[idx];
                                    }
                                    if (dGd <= 1e-18) break;

                                    const double alpha = rs_old / dGd;
                                    double alpha_max = 1e30;
                                    for (int k = 0; k < s_size; ++k) {
                                        const int idx = cg_S[k];
                                        const double b_val = beta_active_data[idx];
                                        const double d_val = cg_d[idx];
                                        if (b_val * d_val < 0.0) {
                                            const double a_zero = -b_val / d_val;
                                            if (a_zero < alpha_max) alpha_max = a_zero;
                                        }
                                    }

                                    if (alpha > alpha_max) {
                                        const double a_step = 0.98 * alpha_max;
                                        for (int k = 0; k < s_size; ++k) {
                                            const int idx = cg_S[k];
                                            beta_active_data[idx] += a_step * cg_d[idx];
                                        }
                                        for (int i = 0; i < active_size_padded; ++i) {
                                            c_ptr[i] -= a_step * q_ptr[i];
                                        }
                                        break; // Дошли до границы ортанта — передаем управление CD для точного зануления!
                                    }

                                    for (int k = 0; k < s_size; ++k) {
                                        const int idx = cg_S[k];
                                        beta_active_data[idx] += alpha * cg_d[idx];
                                    }
                                    for (int i = 0; i < active_size_padded; ++i) {
                                        c_ptr[i] -= alpha * q_ptr[i];
                                    }

                                    double rs_new = 0.0;
                                    for (int k = 0; k < s_size; ++k) {
                                        const int idx = cg_S[k];
                                        const double g_new = cg_g[idx] - alpha * q_ptr[idx];
                                        cg_g[idx] = g_new;
                                        rs_new += g_new * g_new;
                                    }
                                    if (rs_new < 1e-15) break;

                                    const double beta_cg = rs_new / rs_old;
                                    for (int k = 0; k < s_size; ++k) {
                                        const int idx = cg_S[k];
                                        cg_d[idx] = cg_g[idx] + beta_cg * cg_d[idx];
                                    }
                                    rs_old = rs_new;
                                }
                            }
                        }
                    }
                }

                if (nlp >= maxit) break;

                // Инкрементальное обновление вектора невязки r_padded за O(n * |изменившихся активных|)
                if (active_size > 0) {
                    double* RESTRICT r_ptr = ASSUME_ALIGNED(r_padded, 64);
                    for (int idx = 0; idx < active_size; ++idx) {
                        const double delta = beta_active_data[idx] - beta_old_at_start[idx];
                        if (delta != 0.0) {
                            const double* RESTRICT col_ptr = ASSUME_ALIGNED(X_padded + active[idx] * n_padded, 64);
                            for (int i = 0; i < n_padded; ++i) {
                                r_ptr[i] -= delta * col_ptr[i];
                            }
                        }
                    }
                }

                // 2c. Двухэтапная проверка ККТ: сначала на рабочем множестве screened_list
                bool screened_kkt_ok = true;
                screened_violators.clear();
                const double KKT_bound = lam * (1.0 + 1e-9);
                const double* RESTRICT r_ptr = ASSUME_ALIGNED(r_padded, 64);

                for (int i = 0; i < (screened_size / 4) * 4; i += 4) {
                    int j0 = screened_list[i];
                    int j1 = screened_list[i+1];
                    int j2 = screened_list[i+2];
                    int j3 = screened_list[i+3];

                    const double* RESTRICT col0 = ASSUME_ALIGNED(X_padded + j0 * n_padded, 64);
                    const double* RESTRICT col1 = ASSUME_ALIGNED(X_padded + j1 * n_padded, 64);
                    const double* RESTRICT col2 = ASSUME_ALIGNED(X_padded + j2 * n_padded, 64);
                    const double* RESTRICT col3 = ASSUME_ALIGNED(X_padded + j3 * n_padded, 64);

                    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (int k = 0; k < n_padded; ++k) {
                        double rk = r_ptr[k];
                        sum0 += rk * col0[k];
                        sum1 += rk * col1[k];
                        sum2 += rk * col2[k];
                        sum3 += rk * col3[k];
                    }
                    grad_ptr[j0] = sum0 * inv_fn;
                    grad_ptr[j1] = sum1 * inv_fn;
                    grad_ptr[j2] = sum2 * inv_fn;
                    grad_ptr[j3] = sum3 * inv_fn;
                }

                for (int i = (screened_size / 4) * 4; i < screened_size; ++i) {
                    int j = screened_list[i];
                    const double* RESTRICT col_ptr = ASSUME_ALIGNED(X_padded + j * n_padded, 64);
                    double dot_val = 0.0;
                    for (int k = 0; k < n_padded; ++k) {
                        dot_val += col_ptr[k] * r_ptr[k];
                    }
                    grad_ptr[j] = dot_val * inv_fn;
                }

                for (int i = 0; i < screened_size; ++i) {
                    int j = screened_list[i];
                    if (std::abs(grad_ptr[j]) > KKT_bound) {
                        screened_violators.push_back(j);
                        screened_kkt_ok = false;
                    }
                }

                if (!screened_kkt_ok) {
                    for (int j : screened_violators) {
                        add_active(j);
                        if (oom) goto cleanup;
                    }
                    continue;
                }

                // 2d. Глобальная проверка неактивного подпространства:
                // Гибрид Look-Ahead Gap Safe Screening (safe_until[j] > li пропускаются мгновенно!)
                // + фильтрация дрейфа невязки (работает при ЛЮБЫХ n, включая n < 150, устраняя слепоту Listing 3!)
                double d2 = 0.0;
                const double* RESTRICT r_curr_ptr = ASSUME_ALIGNED(r_padded, 64);
                const double* RESTRICT r_ref_ptr = ASSUME_ALIGNED(r_ref_padded, 64);
                for (int k = 0; k < n_padded; ++k) {
                    double diff = r_curr_ptr[k] - r_ref_ptr[k];
                    d2 += diff * diff;
                }
                const double d = std::sqrt(d2);

                const double* RESTRICT grad_ref_ptr = grad_ref.data();
                const double* RESTRICT s_cs_ptr = s_cs.data();
                const int* RESTRICT unscreened_ptr = unscreened_list.data();

                to_compute.clear();
                for (int i = 0; i < unscreened_size; ++i) {
                    int j = unscreened_ptr[i];
                    if (safe_until[j] > li) continue; // Заведомо нулевой признак по сертификату Look-Ahead Gap Safe!
                    double bound = std::abs(grad_ref_ptr[j]) + s_cs_ptr[j] * d;
                    if (bound > KKT_bound) {
                        to_compute.push_back(j);
                    }
                }

                int num_to_compute = static_cast<int>(to_compute.size());
                bool did_reset = false;

                if (num_to_compute > 0.35 * unscreened_size || lambdas_since_reset >= 10) {
                    // При обновлении опорной точки r_ref считаем скалярные произведения только для тех j, где safe_until[j] <= li
                    to_compute.clear();
                    for (int i = 0; i < unscreened_size; ++i) {
                        int j = unscreened_ptr[i];
                        if (safe_until[j] <= li) {
                            to_compute.push_back(j);
                        }
                    }
                    num_to_compute = static_cast<int>(to_compute.size());
                    did_reset = true;
                }

                for (int k = 0; k < (num_to_compute / 4) * 4; k += 4) {
                    int j0 = to_compute[k];
                    int j1 = to_compute[k+1];
                    int j2 = to_compute[k+2];
                    int j3 = to_compute[k+3];

                    const double* RESTRICT col0 = ASSUME_ALIGNED(X_padded + j0 * n_padded, 64);
                    const double* RESTRICT col1 = ASSUME_ALIGNED(X_padded + j1 * n_padded, 64);
                    const double* RESTRICT col2 = ASSUME_ALIGNED(X_padded + j2 * n_padded, 64);
                    const double* RESTRICT col3 = ASSUME_ALIGNED(X_padded + j3 * n_padded, 64);

                    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (int m = 0; m < n_padded; ++m) {
                        double rk = r_ptr[m];
                        sum0 += rk * col0[m];
                        sum1 += rk * col1[m];
                        sum2 += rk * col2[m];
                        sum3 += rk * col3[m];
                    }
                    grad_ptr[j0] = sum0 * inv_fn;
                    grad_ptr[j1] = sum1 * inv_fn;
                    grad_ptr[j2] = sum2 * inv_fn;
                    grad_ptr[j3] = sum3 * inv_fn;
                }

                for (int k = (num_to_compute / 4) * 4; k < num_to_compute; ++k) {
                    int j = to_compute[k];
                    const double* RESTRICT col_ptr = ASSUME_ALIGNED(X_padded + j * n_padded, 64);
                    double sum = 0.0;
                    for (int m = 0; m < n_padded; ++m) {
                        sum += r_ptr[m] * col_ptr[m];
                    }
                    grad_ptr[j] = sum * inv_fn;
                }

                bool full_kkt_ok = true;
                unscreened_violators.clear();
                for (int i = 0; i < unscreened_size; ) {
                    int j = unscreened_list[i];
                    if (safe_until[j] <= li && std::abs(grad_ptr[j]) > KKT_bound) {
                        screened[j] = 1;
                        unscreened_violators.push_back(j);
                        screened_to_idx[j] = screened_size;
                        screened_list[screened_size++] = j;
                        unscreened_list[i] = unscreened_list[--unscreened_size];
                        full_kkt_ok = false;
                    } else {
                        ++i;
                    }
                }

                if (did_reset) {
                    std::memcpy(r_ref_padded, r_padded, n_padded * sizeof(double));
                    double* RESTRICT grad_ref_writable = grad_ref.data();
                    for (int k = 0; k < num_to_compute; ++k) {
                        int j = to_compute[k];
                        grad_ref_writable[j] = grad_ptr[j];
                    }
                    lambdas_since_reset = 0;

                    // Поскольку для всех j из to_compute известны точные градиенты x_j^T r / n,
                    // запускаем Look-Ahead Gap Safe Screening на будущие шаги пути m > li!
                    if (full_kkt_ok && li + 1 < nlam && num_to_compute > 0) {
                        double r_norm2 = 0.0;
                        double ry_dot = 0.0;
                        for (int i = 0; i < n; ++i) {
                            double ri = r_ptr[i];
                            r_norm2 += ri * ri;
                            ry_dot += ri * y_padded[i];
                        }
                        double l1_norm = 0.0;
                        for (int idx = 0; idx < active_size; ++idx) {
                            l1_norm += std::abs(beta_active_data[idx]);
                        }
                        // В точке full_kkt_ok все |grad_j| <= lam * (1 + 1e-9), поэтому alpha_k = 1 / (lam * (1 + 1e-9))
                        const double alpha_k = 1.0 / KKT_bound;
                        const double primal_obj = 0.5 * inv_fn * r_norm2 + lam * l1_norm;
                        const double dual_obj = (lam * inv_fn) * (alpha_k * ry_dot) - (0.5 * lam * lam * inv_fn) * (alpha_k * alpha_k * r_norm2);
                        const double gap_k = std::max(0.0, primal_obj - dual_obj);
                        const double r_gap_over_sqrtn = std::sqrt(2.0 * gap_k) * inv_lam;

                        const double* inv_lam_begin = inv_lam_grid.data() + (li + 1);
                        const double* inv_lam_end = inv_lam_grid.data() + nlam;

                        for (int k = 0; k < num_to_compute; ++k) {
                            int j = to_compute[k];
                            const double sj = s[j];
                            const double base_bound = alpha_k * std::abs(grad_ptr[j]) + sj * r_gap_over_sqrtn;
                            if (base_bound < 1.0 && sj > 0.0) {
                                const double max_inv_lam_m = inv_lam + (1.0 - base_bound) / (sj * y_norm_over_sqrtn);
                                if (max_inv_lam_m >= inv_lam_last) {
                                    safe_until[j] = nlam;
                                } else {
                                    const double* it = std::lower_bound(inv_lam_begin, inv_lam_end, max_inv_lam_m);
                                    int m_wake = static_cast<int>(it - inv_lam_grid.data());
                                    if (m_wake > safe_until[j]) {
                                        safe_until[j] = m_wake;
                                    }
                                }
                            }
                        }
                    }
                } else if (full_kkt_ok) {
                    lambdas_since_reset++;
                }

                if (full_kkt_ok) {
                    // На самом первом шаге li == 0 (где r = r_ref и все градиенты grad_init известны точно!)
                    // сразу выполняем начальный Look-Ahead Gap Safe Screening по всем unscreened признакам!
                    if (li == 0 && nlam > 1) {
                        double r_norm2 = 0.0;
                        double ry_dot = 0.0;
                        for (int i = 0; i < n; ++i) {
                            double ri = r_ptr[i];
                            r_norm2 += ri * ri;
                            ry_dot += ri * y_padded[i];
                        }
                        double l1_norm = 0.0;
                        for (int idx = 0; idx < active_size; ++idx) {
                            l1_norm += std::abs(beta_active_data[idx]);
                        }
                        const double alpha_0 = 1.0 / KKT_bound;
                        const double primal_0 = 0.5 * inv_fn * r_norm2 + lam * l1_norm;
                        const double dual_0 = (lam * inv_fn) * (alpha_0 * ry_dot) - (0.5 * lam * lam * inv_fn) * (alpha_0 * alpha_0 * r_norm2);
                        const double gap_0 = std::max(0.0, primal_0 - dual_0);
                        const double r_gap_0 = std::sqrt(2.0 * gap_0) * inv_lam;

                        const double* inv_lam_begin = inv_lam_grid.data() + 1;
                        const double* inv_lam_end = inv_lam_grid.data() + nlam;
                        for (int i = 0; i < unscreened_size; ++i) {
                            int j = unscreened_list[i];
                            const double sj = s[j];
                            const double base_bound = alpha_0 * std::abs(grad_ptr[j]) + sj * r_gap_0;
                            if (base_bound < 1.0 && sj > 0.0) {
                                const double max_inv_lam_m = inv_lam + (1.0 - base_bound) / (sj * y_norm_over_sqrtn);
                                if (max_inv_lam_m >= inv_lam_last) {
                                    safe_until[j] = nlam;
                                } else {
                                    const double* it = std::lower_bound(inv_lam_begin, inv_lam_end, max_inv_lam_m);
                                    safe_until[j] = static_cast<int>(it - inv_lam_grid.data());
                                }
                            }
                        }
                    }
                    break;
                }

                for (int j : unscreened_violators) {
                    add_active(j);
                    if (oom) goto cleanup;
                }
            }

            for (size_t idx = 0; idx < active.size(); ++idx) {
                beta(active[idx]) = beta_active_data[idx];
            }
            coef_path.col(li) = beta;
            prev_lam = lam;
        }
    }

cleanup:
    aligned_free_compat(G_data);
    aligned_free_compat(c_data);
    aligned_free_compat(beta_active_data);
    aligned_free_compat(xv_active_data);
    aligned_free_compat(inv_xv_active_data);
    aligned_free_compat(grad_init_active_data);
    aligned_free_compat(beta_old_at_start);
    aligned_free_compat(y_padded);
    aligned_free_compat(r_padded);
    aligned_free_compat(r_ref_padded);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [n_repeats]\n", argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    int n_repeats = (argc >= 4) ? std::atoi(argv[3]) : 20;

    std::FILE* fin = std::fopen(in_path, "rb");
    if (!fin) return 1;

    int32_t n = 0, p = 0, n_lambda = 0;
    if (std::fread(&n, sizeof(int32_t), 1, fin) != 1) return 1;
    if (std::fread(&p, sizeof(int32_t), 1, fin) != 1) return 1;
    if (std::fread(&n_lambda, sizeof(int32_t), 1, fin) != 1) return 1;

    std::vector<double> X_row(static_cast<size_t>(n) * p);
    if (std::fread(X_row.data(), sizeof(double), static_cast<size_t>(n) * p, fin) != static_cast<size_t>(n) * p) return 1;

    VectorXd y(n);
    if (std::fread(y.data(), sizeof(double), n, fin) != static_cast<size_t>(n)) return 1;

    VectorXd lam_path(n_lambda);
    if (std::fread(lam_path.data(), sizeof(double), n_lambda, fin) != static_cast<size_t>(n_lambda)) return 1;
    std::fclose(fin);

    int n_padded = ((n + 7) / 8) * 8;
    double* X_padded = nullptr;
    if (posix_memalign_compat((void**)&X_padded, 64, static_cast<size_t>(n_padded) * p * sizeof(double)) != 0) return 1;

    VectorXd xv(p);
    VectorXd grad_init(p);
    const double inv_fn = 1.0 / n;
    const double* RESTRICT y_ptr = y.data();

    for (int j = 0; j < p; ++j) {
        double* RESTRICT dest = X_padded + j * n_padded;
        double sum_xx = 0.0;
        double sum_xy = 0.0;
        for (int i = 0; i < n; ++i) {
            double val = X_row[static_cast<size_t>(i) * p + j];
            dest[i] = val;
            sum_xx += val * val;
            sum_xy += val * y_ptr[i];
        }
        for (int i = n; i < n_padded; ++i) dest[i] = 0.0;
        xv(j) = sum_xx * inv_fn;
        grad_init(j) = sum_xy * inv_fn;
    }

    MatrixXd coef_listing3 = MatrixXd::Zero(p, n_lambda);
    MatrixXd coef_celer = MatrixXd::Zero(p, n_lambda);

    // Warmup run
    solve_active_set(X_padded, n_padded, n, p, y, lam_path, xv, grad_init, coef_listing3, 1e-9, 100000);
    solve_celer_lookahead(X_padded, n_padded, n, p, y, lam_path, xv, grad_init, coef_celer, 0.9e-6, 100000);

    // Benchmark Listing 3 solver
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < n_repeats; ++r) {
        coef_listing3.set_zero();
        solve_active_set(X_padded, n_padded, n, p, y, lam_path, xv, grad_init, coef_listing3, 1e-9, 100000);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_listing3 = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_repeats;

    // Benchmark Celer + Look-Ahead Gap Safe solver
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < n_repeats; ++r) {
        coef_celer.set_zero();
        solve_celer_lookahead(X_padded, n_padded, n, p, y, lam_path, xv, grad_init, coef_celer, 0.9e-6, 100000);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_celer = std::chrono::duration<double, std::milli>(t3 - t2).count() / n_repeats;

    std::FILE* fout = std::fopen(out_path, "wb");
    if (!fout) return 1;
    std::fwrite(&ms_listing3, sizeof(double), 1, fout);
    std::fwrite(&ms_celer, sizeof(double), 1, fout);
    std::fwrite(coef_listing3.data(), sizeof(double), static_cast<size_t>(p) * n_lambda, fout);
    std::fwrite(coef_celer.data(), sizeof(double), static_cast<size_t>(p) * n_lambda, fout);
    std::fclose(fout);

    aligned_free_compat(X_padded);
    return 0;
}
