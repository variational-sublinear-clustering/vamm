/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
#include <pybind11/functional.h>
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <algorithm>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <iostream>
#include <list>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Numpy.h"
#include "checks.h"
#include "omp_helper.h"

#ifdef _OPENMP
#include <omp.h>
#endif

template <class key_t = size_t, class val_t = precision_t>
using map_t = boost::unordered::unordered_flat_map<key_t, val_t>;

template <class key_t = size_t>
using set_t = boost::unordered::unordered_flat_set<key_t>;

// datatype to store truncated variational distribution
using q_t = std::vector<std::pair<size_t, precision_t>>;
using q_t_it = std::vector<std::pair<size_t, q_t::iterator>>;

struct Triplet {
    size_t c;
    size_t num_ljs;
    precision_t sum_ljs;

    Triplet() {
        c = 0;
        num_ljs = 0;
        sum_ljs = 0.;
    }
    Triplet(size_t c_, size_t num_ljs_, precision_t sum_ljs_) : c(c_), num_ljs(num_ljs_), sum_ljs(sum_ljs_) {}
};

class Variational {
   public:
    static constexpr precision_t infty_ = std::numeric_limits<precision_t>::infinity();

    size_t N;
    size_t C;
    size_t C_prime;
    size_t G;
    size_t C_relocate;
    size_t E;
    bool M_step_hard;

    std::vector<q_t> qs;
    std::vector<std::vector<q_t>> partition;
    std::vector<std::vector<q_t_it>> partition_it;
    std::vector<std::vector<size_t>> gc_set;
    size_t number_ljs;
    Random<std::mt19937_64> rng;

    size_t initial_seed;

    void get_partition(void);
    void get_partition_it(void);
    void get_partition_hard(void);

    template <typename Partition, typename FillFunc>
    void get_partition_impl(Partition& part, FillFunc fill_func);

    bool relocate_discarded;
    std::vector<size_t> relocate_from;
    std::vector<size_t> relocate_to;

    template <class Model>
    void E_step_construct_Sn(const Model& model);

    template <class Model>
    size_t E_step_ljs(cRef<Matrix<>> X, const Model& model);

    precision_t E_step_normalize(const precision_t&);

    precision_t (*compute_relevance)(const precision_t&, const precision_t&);
    void (*finalize_sum_ljs)(std::vector<Triplet>&, cRef<Vector<>>, const size_t);

    template <class Model>
    void E_step_update_gc_set(const Model& model);

    void E_step_select(void);

    template <class Model>
    void relocate_discarded_components(Model& model);

    void init_shared_(const std::string&);

    Variational(size_t, size_t, size_t, size_t, size_t, size_t, bool, bool, std::string);
    Variational(size_t, size_t, size_t, size_t, size_t, size_t, cRef<Vector<size_t>>, bool, bool, std::string);

    void initialize_full_qs_();
    void initialize_qs_random();
    void initialize_qs_indices(cRef<Vector<size_t>> indices);
    void initialize_gc_set_();
    void initialize_thread_storage_();

    template <class Model>
    precision_t EM_step(cRef<Matrix<>> X, Model& model, const bool fit, const bool update_var_params,
                        const precision_t& beta);

    template <class Model>
    void E_step_ljs_order_n(cRef<Matrix<>> X, const Model& model);

    template <class Model>
    void E_step_ljs_order_c(cRef<Matrix<>> X, const Model& model);

    template <class Model>
    precision_t E_step(cRef<Matrix<>> X, Model& model, const bool update_var_params, const precision_t& beta);

    template <class Model>
    void M_step(cRef<Matrix<>> X, Model& model);

    std::unordered_map<size_t, precision_t> q_map(size_t n) const;

    void q_in(size_t n, const std::unordered_map<size_t, precision_t>& map);

    auto approx_map(size_t n) const;

    auto q_to_sparse_matrix(void) const;
    void q_from_sparse_matrix(const SparseMatrix<> sp_mat);

    auto q_shrinked_to_sparse_matrix(cRef<Vector<bool>> Mask) const;

    auto get_gc_set(const size_t c);

    Vector<size_t> indices(void) const;

    auto gc_set_to_sparse_matrix(void) const;
    void gc_set_from_sparse_matrix(const SparseMatrix<> sp_mat);

    // template <class Model>
    // Vector<> marginal_probabilities(cRef<Matrix<>> X, Model& model);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
    template <typename... Model>
    static void bind(py::module_& m);
#endif
};

//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Computes the pairwise Kullback–Leibler relevance measure.
 *
 * Returns the difference between two log-probability values. This quantity is
 * used as a relevance score when comparing components during gc set
 * update.
 *
 * @param log_prob1 Log-probability of the reference component.
 * @param log_prob2 Log-probability of the candidate component.
 *
 * @return Difference `log_prob1 - log_prob2`.
 */
precision_t compute_KL_divergence(const precision_t& log_prob1, const precision_t& log_prob2) {
    return log_prob1 - log_prob2;
}

/**
 * @brief Computes a Euclidean-based relevance measure.
 *
 * Returns the negated log-probability of the candidate component. The first
 * argument is unused but retained to provide the same interface as other
 * relevance functions.
 *
 * @param Unused reference log-probability.
 * @param log_prob Log-probability of the candidate component.
 *
 * @return Negated log-probability `-log_prob`.
 */
precision_t compute_euclidean(const precision_t&, const precision_t& log_prob) { return -log_prob; }

void norm(std::vector<Triplet>& candidates, cRef<Vector<>>, const size_t) {
    for (auto& it : candidates) {
        it.sum_ljs /= it.num_ljs;
    }
}

void remove_prior_norm(std::vector<Triplet>& candidates, cRef<Vector<>> P_log, const size_t c) {
    for (auto& it : candidates) {
        it.sum_ljs /= it.num_ljs;
        it.sum_ljs += P_log[it.c] - P_log[c];
    }
}

/**
 * @brief Normalizes the variational distribution of a single data point.
 *
 * The log-joint values stored in `qs[n]` are transformed into normalized
 * posterior probabilities using a numerically stable softmax operation with
 * inverse temperature `beta` (default 1.0). The maximum log-joint value is subtracted before
 * exponentiation to improve numerical stability.
 *
 * The q-values in `qs[n]` are replaced in-place by their normalized
 * probabilities.
 *
 * @param n Index of the data point.
 * @param beta Inverse temperature used for annealing.
 *
 * @return A pair consisting of:
 *         - the normalization constant before taking the logarithm, and
 *         - the shifted maximum log-joint value (`beta * max_log_joint`).
 *
 * @throws std::runtime_error If the maximum log-joint value is not finite.
 */
void normalize(q_t& q, const precision_t& beta, precision_t& lim, precision_t& sum) {
    lim = std::max_element(q.begin(), q.end(), [](auto& lhs, auto& rhs) -> bool {
              return lhs.second < rhs.second;
          })->second;
    lim *= beta;
    sum = 0;
    for (auto& [c, q_nc] : q) {
        q_nc = std::exp(beta * q_nc - lim);
        sum += q_nc;
    }
    if (!std::isfinite(sum)) {
        throw std::runtime_error("NaN in logjoints!");
    }
    for (auto& [c, q_nc] : q) {
        q_nc /= sum;
    }
}

/**
 * @brief Constructs a Variational object with randomly initialized Kn sets.
 *
 * Initializes the variational model with N data points, C components,
 * size of the Kn set C_prime, and G components considered for each
 * component update.
 *
 * The constructor initializes all internal storage, validates the input
 * parameters, configures the similarity measure, creates the gc_sets,
 * initializes thread-local storage, and randomly initializes the Kn sets.
 *
 * @param _N Number of data points.
 * @param _C Total number of components.
 * @param _C_prime Size of the Kn sets.
 * @param _G Number of components considered during local updates.
 * @param _E Number of randomly drawn components per data point.
 * @param _seed Random seed used for stochastic initialization.
 * @param _relocate_discarded Whether discarded components should be relocated.
 * @param _hard Whether hard assignments should be used during the M-step.
 * @param _sim_measure Similarity measure used for relevance computation.
 *        Supported values include "Euclidean" and "KL".
 */
Variational::Variational(size_t _N, size_t _C, size_t _C_prime, size_t _G, size_t _E, size_t _seed,
                         bool _relocate_discarded = true, bool _hard = false, std::string _sim_measure = "") :
    N(_N),
    C(_C),
    C_prime(_C_prime),
    G(_G),
    C_relocate(0),
    E(_E),
    M_step_hard(_hard),
    qs(_N),
    partition(get_max_threads()),
    partition_it(get_max_threads()),
    gc_set(_C),
    number_ljs(0),
    rng(_seed),
    initial_seed(_seed),
    relocate_discarded(_relocate_discarded),
    relocate_from(),
    relocate_to() {
    init_shared_(_sim_measure);
    initialize_qs_random();
}

/**
 * @brief Constructs a Variational object using predefined initial component assignments.
 *
 * Similar to the default constructor, but initializes the variational
 * assignments using the provided component indices before filling remaining
 * assignments randomly if C_prime active components per data point are required.
 *
 * The size of @p indices determines its interpretation:
 * - If its size equals N, each entry specifies the initial component assignment
 *   of the corresponding data point.
 * - If its size equals C, each entry specifies the data point assigned to the
 *   corresponding component.
 *
 * @param _N Number of data points.
 * @param _C Total number of components.
 * @param _C_prime Size of the Kn sets.
 * @param _G Number of components considered during local updates.
 * @param _E Number of randomly drawn components per data point.
 * @param _seed Random seed used for stochastic initialization.
 * @param indices Initial component assignments.
 * @param _relocate_discarded Whether discarded components should be relocated.
 * @param _hard Whether hard assignments should be used during the M-step.
 * @param _sim_measure Similarity measure used for relevance computation.
 *        Supported values include "Euclidean" and "KL".
 */
Variational::Variational(size_t _N, size_t _C, size_t _C_prime, size_t _G, size_t _E, size_t _seed,
                         cRef<Vector<size_t>> indices, bool _relocate_discarded = true, bool _hard = false,
                         std::string _sim_measure = "") :
    N(_N),
    C(_C),
    C_prime(_C_prime),
    G(_G),
    C_relocate(0),
    E(_E),
    M_step_hard(_hard),
    qs(_N),
    partition(get_max_threads()),
    partition_it(get_max_threads()),
    gc_set(_C),
    number_ljs(0),
    rng(_seed),
    initial_seed(_seed),
    relocate_discarded(_relocate_discarded),
    relocate_from(),
    relocate_to() {
    init_shared_(_sim_measure);
    initialize_qs_indices(indices);
    initialize_qs_random();
}

/**
 * @brief Performs initialization shared by all constructors.
 *
 * Validates model parameters, selects the similarity and normalization
 * functions for the update of the gc sets, initializes qs if C_prime = C,
 * creates the gc sets, and allocates thread-local storage.
 *
 * @param _sim_measure Similarity measure used for relevance computation.
 *
 * @throws std::invalid_argument If model parameters violate required bounds.
 */
void Variational::init_shared_(const std::string& _sim_measure = "") {
    if (N <= 0) {
        throw std::invalid_argument("N must be > 0");
    }
    if (C <= 0) {
        throw std::invalid_argument("C must be > 0");
    }
    if ((C_prime <= 0) || (C_prime > C)) {
        throw std::invalid_argument("1 <= C_prime <= C must hold");
    }
    if ((G <= 0) || (G > C)) {
        throw std::invalid_argument("1 <= G <= C must hold");
    }

    if (_sim_measure == "Euclidean") {
        compute_relevance = compute_euclidean;
        finalize_sum_ljs = norm;
    } else {
        compute_relevance = compute_KL_divergence;
        finalize_sum_ljs = remove_prior_norm;
    }

    if (C_prime == C) {
        initialize_full_qs_();
    }

    initialize_gc_set_();
    initialize_thread_storage_();
}

/**
 * @brief Initializes Kn sets with all components when C_prime equals C.
 *
 * Each data point receives an assignment to every component with initial weight
 * 1.0.
 */
void Variational::initialize_full_qs_() {
#pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        qs[n].reserve(C_prime * G + E);

        for (size_t c = 0; c < C; ++c) {
            qs[n].emplace_back(c, 1.0);
        }
    }
}

/**
 * @brief Initializes Kn sets randomly.
 *
 * Ensures that the Kn set of each data point has exactly C_prime components.
 * Existing assignments are preserved as initial candidates and completed with
 * randomly sampled components if necessary.
 *
 * This function has no effect if C_prime equals C, since assignments are then
 * already initialized densely.
 */
void Variational::initialize_qs_random() {
    if (C_prime == C) return;

#pragma omp parallel
    {
        set_t<> set;
        std::uniform_int_distribution<size_t> rand_int(0, C - 1);

        set.reserve(C_prime);

#pragma omp for
        for (size_t n = 0; n < N; ++n) {
            qs[n].reserve(C_prime * G + E);

            // check if qs[n] is already initalized with indices
            for (auto& [c, _] : qs[n]) {
                set.insert(c);
            }

            while (set.size() < C_prime) set.insert(rand_int(rng()));

            qs[n].clear();
            for (auto c : set) qs[n].emplace_back(c, 1.0);

            set.clear();
        }
    }
}

/**
 * @brief Initializes Kn sets from user-provided indices.
 *
 * The input can describe assignments in two different formats:
 *
 * - Size N: indices[n] specifies the initial component of data point n.
 * - Size C: indices[c] specifies the data point associated with component c.
 *
 * Additional components are later added by initialize_qs_random().
 *
 * @param indices Initial component assignment information.
 *
 * @throws std::invalid_argument If indices has neither size N nor size C.
 */
void Variational::initialize_qs_indices(cRef<Vector<size_t>> indices) {
    if (C_prime == C) return;

    const size_t size = indices.size();

    if (size != N && size != C) throw std::invalid_argument("indices must be of size N or C");

    if (size == C) {
        for (size_t c = 0; c < C; ++c) {
            qs[indices[c]].emplace_back(c, 1.);
        }
    } else {
#pragma omp parallel for
        for (size_t n = 0; n < N; ++n) {
            qs[n].reserve(C_prime * G + E);
            qs[n].emplace_back(indices[n], 1.);
        }
    }
}

/**
 * @brief Initializes the gc_set.
 *
 * For each component, selects G related components. If G equals C, every component
 * is connected to every other component. Otherwise, the remaining neighbors are
 * sampled randomly.
 *
 */
void Variational::initialize_gc_set_() {
#pragma omp parallel
    {
        set_t<> set;
        std::uniform_int_distribution<size_t> rand_int(0, C - 1);

        set.reserve(G);

#pragma omp for
        for (size_t c = 0; c < C; ++c) {
            gc_set[c].reserve(G);

            if (G == C) {
                for (size_t g = 0; g < G; ++g) gc_set[c].emplace_back(g);
            } else {
                set.insert(c);

                while (set.size() < G) set.insert(rand_int(rng()));

                for (auto g : set) gc_set[c].emplace_back(g);
            }

            set.clear();
        }
    }
}

/**
 * @brief Initializes thread-local storage for parallel computations.
 *
 * Allocates per-thread partition buffers and iterators used by OpenMP-based
 * update routines.
 */
void Variational::initialize_thread_storage_() {
#pragma omp parallel
    {
        size_t thread_num = get_thread_num();

        partition[thread_num] = std::vector<q_t>(C);
        partition_it[thread_num] = std::vector<q_t_it>(C);
    }
}

/**
 * @brief Creates the Sn set (search space) for each data point.
 *
 * For every data point, this function constructs the Sn set of
 * components consisting of:
 * - neighboring components of the currently active components according to
 *   the gc set, and
 * - a fixed number of randomly sampled components to encourage exploration.
 *
 * Only active components enabled by the model mask are considered. The Sn set replaces the previous Kn set
 * of `qs`, and each inserted component is initialized with a placeholder log-joint value of 1.0. These
 * values are overwritten during the subsequent E-step.
 *
 * The expansion is performed independently for each data point and is
 * parallelized using OpenMP.
 *
 * @param model Model instance containing the active component mask.
 *
 * @note Existing log-joint values are discarded during expansion.
 */
template <class Model>
void Variational::E_step_construct_Sn(const Model& model) {
#pragma omp parallel
    {
        size_t c;
        set_t<> set;
        std::uniform_int_distribution<size_t> rand_int(0, C - 1);

#pragma omp for
        for (size_t n = 0; n < N; n++) {
            for (const auto& [k, _] : qs[n]) {
                if (!model.Mask[k]) continue;
                for (const auto g : gc_set[k]) {
                    if (!model.Mask[g]) continue;
                    set.insert(g);
                }
            }
            for (size_t e = 0; e < E; e++) {
                do {
                    c = rand_int(rng());
                } while (!model.Mask[c]);
                set.insert(c);
            }
            qs[n].clear();
            for (const size_t k : set) {
                qs[n].emplace_back(k, 1.);
            }
            set.clear();
        }
    }
}

/**
 * @brief Builds the soft partition of components based on the current Kn/Sn set (used for M_step).
 *
 * This function assigns each datapoint xn to all components c for which a
 * corresponding q-value exists (c is in the respective Kn/Sn set).
 * The partition stores pairs consisting of the datapoint index and its associated log-joint/ posterior.
 *
 * The partition is constructed using get_partition_impl(), which performs the
 * operation in parallel over all datapoints.
 *
 * @note The resulting partition is stored in the member variable `partition`.
 */
void Variational::get_partition(void) {
    get_partition_impl(partition, [this](auto& part, size_t n) {
        for (auto q = qs[n].begin(); q != qs[n].end(); q++) {
            part[q->first].emplace_back(n, q->second);
        }
    });
}

/**
 * @brief Builds a soft partition while storing iterators to q-values (used for E_step_ljs_order_c).
 *
 * Similar to get_partition(), this function assigns datapoints to all components
 * with an associated q-value. Instead of storing a copy of the q-value, the
 * partition stores iterators pointing directly to the corresponding entries
 * in the q-value containers.
 *
 * Storing iterators allows direct access to the original data.
 *
 * The construction is parallelized internally by get_partition_impl().
 *
 * @note The resulting partition is stored in the member variable `partition_it`.
 */
void Variational::get_partition_it(void) {
    get_partition_impl(partition_it, [this](auto& part, size_t n) {
        for (auto q = qs[n].begin(); q != qs[n].end(); q++) {
            part[q->first].emplace_back(n, q);
        }
    });
}

/**
 * @brief Builds a hard partition by assigning each datapoint to its most likely component (used for
 * E_step_update_gc).
 *
 * For every datapoint xn, only the component with the largest q-value among the
 * first C_prime entries is selected. The datapoint is then assigned exclusively
 * to this component.
 *
 * This creates a hard partition in contrast to the soft partitions generated
 * by get_partition() and get_partition_it().
 *
 * The construction is parallelized internally by get_partition_impl().
 *
 * @note The resulting partition is stored in the member variable `partition`.
 */
void Variational::get_partition_hard(void) {
    get_partition_impl(partition, [this](auto& part, size_t n) {
        size_t end = std::min(C_prime, qs[n].size());

        auto it = std::max_element(qs[n].begin(), qs[n].begin() + end,
                                   [](auto& lhs, auto& rhs) { return lhs.second < rhs.second; });

        part[it->first].emplace_back(n, it->second);
    });
}

/**
 * @brief Generic helper for constructing partition structures in parallel.
 *
 * This function initializes a partition container for each OpenMP thread and
 * applies a user-provided filling function to assign datapoints to partition
 * entries. After processing, allocated memory is reduced using shrink_to_fit().
 *
 * The function distributes the iteration over all datapoints among available
 * OpenMP threads. Each thread operates on its own temporary partition data,
 * avoiding synchronization overhead during partition construction.
 *
 * @param part Partition structure to be filled.
 * @param fill_func Function object that inserts the contribution of a single
 *                  data point into the partition. It receives a thread-local
 *                  partition and the datapoint index.
 */
template <typename Partition, typename FillFunc>
void Variational::get_partition_impl(Partition& part, FillFunc fill_func) {
#pragma omp parallel
    {
        size_t thread_num = get_thread_num();

        std::fill(part[thread_num].begin(), part[thread_num].end(),
                  typename Partition::value_type::value_type{});

#pragma omp for
        for (size_t n = 0; n < N; n++) {
            fill_func(part[thread_num], n);
        }

        for (size_t c = 0; c < C; c++) {
            part[thread_num][c].shrink_to_fit();
        }
    }
}

/**
 * @brief Performs the E-step by iterating over data points.
 *
 * This implementation computes the log-joint probabilities for each datapoint xn
 * and each component c currently in qs[n] (Sn set). The iteration order is
 * optimized for traversing datapoints.
 *
 * Each OpenMP thread obtains model-specific temporary storage through
 * Model::E_step_allocate(), which is passed to E_step_log_joint() during the
 * computation.
 *
 * @param X Data matrix containing the datapoints.
 * @param model Model instance used to compute log-joint probabilities.
 *
 * @note The q-values in `qs` are updated in-place.
 */
template <class Model>
void Variational::E_step_ljs_order_n(cRef<Matrix<>> X, const Model& model) {
#pragma omp parallel
    {
        model.E_step_allocate([&](auto&... e_step_args) -> auto {
#pragma omp for  // schedule(dynamic,1)
            for (size_t n = 0; n < N; n++) {
                for (auto& [c, log_prob] : qs[n]) {
                    model.E_step_log_joint(X.row(n), c, log_prob, e_step_args...);
                }
            }
        });
    }
}

/**
 * @brief Performs the E-step by iterating over components.
 *
 * This implementation first constructs a partition of the current q-values
 * using get_partition_it(). The partition groups datapoints by their active
 * components, allowing the computation to iterate over components c instead
 * of datapoints n.
 *
 * This ordering can improve cache locality and reduce unnecessary component
 * traversal when the model has many parameters.
 *
 * Each OpenMP thread obtains model-specific temporary storage through
 * Model::E_step_allocate(), which is passed to E_step_log_joint() during the
 * computation.
 *
 * @param X Data matrix containing the datapoints.
 * @param model Model instance used to compute log-joint probabilities.
 *
 * @note The q-values in `qs` are updated through iterators stored in
 *       `partition_it`.
 */
template <class Model>
void Variational::E_step_ljs_order_c(cRef<Matrix<>> X, const Model& model) {
    get_partition_it();
#pragma omp parallel
    {
        model.E_step_allocate([&](auto&... e_step_args) -> auto {
#pragma omp for schedule(dynamic, 1)
            for (std::size_t c = 0; c < C; c++) {
                for (const auto& _partition : partition_it) {
                    if (_partition[c].empty()) continue;

                    for (auto& [n, q] : _partition[c]) {
                        model.E_step_log_joint(X.row(n), c, q->second, e_step_args...);
                    }
                }
            }
        });
    }
}

/**
 * @brief Performs the variational E-step computation of log-joints.
 *
 * The function selects the most efficient E-step iteration strategy depending
 * on the model configuration. If Model::loop_order_n is enabled, the
 * computation iterates over datapoints; otherwise, it iterates over components
 * using a precomputed partition.
 *
 * After computing the log-joint probabilities, the function counts the total
 * number of evaluated log-joint values and prunes the q-distribution
 * of each data point. If more than C_prime components are present, only the
 * C_prime highest-scoring components are retained using std::nth_element.
 *
 * @param X Data matrix containing the datapoints.
 * @param model Model instance used for computing log-joint probabilities.
 *
 * @return Number of log-joint probability values evaluated before pruning.
 *
 * @note The ordering of elements in qs[n] after pruning is unspecified due to
 *       the use of std::nth_element.
 */
template <class Model>
size_t Variational::E_step_ljs(cRef<Matrix<>> X, const Model& model) {
    if constexpr (Model::loop_order_n) {
        E_step_ljs_order_n(X, model);
    } else {
        E_step_ljs_order_c(X, model);
    }
    size_t total_ljs = 0;

#pragma omp parallel for
    for (size_t n = 0; n < N; n++) {
#pragma omp atomic
        total_ljs += qs[n].size();

        if (qs[n].size() > C_prime) {
            std::nth_element(qs[n].begin(), qs[n].begin() + C_prime, qs[n].end(),
                             [](auto& lhs, auto& rhs) -> bool { return lhs.second > rhs.second; });
        }
    }

    return total_ljs;
}

/**
 * @brief Updates the gc set.
 *
 * For each active component, this function constructs the gc set of up to
 * G components. The resulting gc set is later used by E_step_construct_Sn() to propose new candidate
 * components during the variational E-step.
 *
 * The procedure consists of the following steps:
 * - Compute a hard partition of the data points using get_partition_hard().
 * - For each component c, collect all components that co-occur with c in the
 *   current search spaces of the assigned data points.
 * - Accumulate a relevance score ('Euclidean' or 'KL' measure) for each candidate neighbor based on the
 *   corresponding log-joint probabilities.
 * - Finalize the relevance scores using finalize_sum_ljs().
 * - Select the G - 1 most relevant neighbors and store them together with c
 *   itself in gc_set[c] (gc set).
 *
 * Components disabled by the model mask are ignored throughout the
 * computation.
 *
 * If G is equal to 1 or C, no update is necessary since the neighborhood gc_set
 * is fixed:
 * - G == 1: every component is only connected to itself.
 * - G == C: every component is connected to all components.
 *
 * The gc_set construction is parallelized over components using OpenMP.
 *
 * @param model Model instance supplying the component mask (`Mask`) and
 *              logarithmic prior probabilities (`P_log`).
 *
 * @note Existing gc_set connections are discarded and rebuilt from scratch.
 * @note The relevance metric is computed by compute_relevance() and finalized
 *       by finalize_sum_ljs().
 */
template <class Model>
void Variational::E_step_update_gc_set(const Model& model) {
    if ((G == 1) or (G == C)) {
        return;  // no update needed
    }
    /* */
    get_partition_hard();
/* */
#pragma omp parallel
    {
        size_t end;
        std::uniform_int_distribution<size_t> rand_int(0, C - 1);
        map_t<size_t, size_t> index_map;
        // vector over replacement candidates: (c, number_of_log_joints, sum_of_log_joints)
        std::vector<Triplet> candidates;
        candidates.reserve(C);
/* */
#pragma omp for schedule(dynamic)
        for (size_t c = 0; c < C; c++) {
            if (!model.Mask[c]) continue;
            gc_set[c].clear();
            gc_set[c].push_back(c);

            for (const auto& list : partition) {
                if (list[c].empty()) continue;

                for (const auto& [n, log_prob_c] : list[c]) {
                    for (const auto& [k, log_prob] : qs[n]) { /* contains search spaces */

                        if (k == c) continue;
                        auto el = index_map.find(k);

                        if (el != index_map.end()) {
                            candidates[el->second].num_ljs++;
                            candidates[el->second].sum_ljs += compute_relevance(log_prob_c, log_prob);
                        } else {
                            index_map.emplace(k, candidates.size());
                            candidates.emplace_back(k, 1, compute_relevance(log_prob_c, log_prob));
                        }
                    }
                }
            }
            if (index_map.empty()) continue;
            finalize_sum_ljs(candidates, model.P_log, c);
            if (index_map.size() >= G) {
                std::nth_element(candidates.begin(), candidates.begin() + (G - 1), candidates.end(),
                                 [](auto& lhs, auto& rhs) -> bool { return lhs.sum_ljs < rhs.sum_ljs; });
            }
            end = std::min(G - 1, index_map.size());
            for (auto it = candidates.begin(); it != candidates.begin() + end; ++it) {
                gc_set[c].push_back(it->c);
            }
            candidates.clear();
            index_map.clear();
        }
    }
    return;
}

/**
 * @brief Truncates the variational distributions to the size of the Kn set (C_prime).
 *
 * For each data point, only the first C_prime components are retained. This
 * function assumes that the components have already been ranked (in E_step_ljs()) such that the
 * most relevant entries appear first.
 *
 * The truncation is performed independently for each data point and is
 * parallelized using OpenMP.
 */
void Variational::E_step_select(void) {
#pragma omp parallel for
    for (size_t n = 0; n < N; n++) {
        if (qs[n].size() > C_prime) {
            qs[n].resize(C_prime);
        }
    }
}

/**
 * @brief Normalizes all variational distributions and evaluates the variational free energy.
 *
 * This function applies normalize() to every data point, replacing the stored
 * log-joint values by normalized posterior probabilities.
 *
 * During normalization, the contribution of each data point to the variational
 * free energy is accumulated. If hard M-step optimization is enabled
 * (`M_step_hard == true`), the logarithmic normalization term is omitted.
 *
 * The normalization is parallelized using OpenMP.
 *
 * @param beta Inverse temperature used for annealing.
 *
 * @return Value of the variational objective after normalization.
 */
precision_t Variational::E_step_normalize(const precision_t& beta) {
    precision_t objective = 0;
#pragma omp parallel
    {
        precision_t objective_thread = 0;
        precision_t sum, lim;
#pragma omp for
        for (std::size_t n = 0; n < N; n++) {
            normalize(qs[n], beta, lim, sum);

            objective_thread += lim;
            objective_thread += M_step_hard ? 0.0 : std::log(sum);
        }
#pragma omp atomic
        objective += objective_thread;
    }
    return objective;
}

/**
 * @brief Performs the variational E-step.
 *
 * The E-step updates the variational posterior distributions over the latent
 * variables. Depending on the configuration, this includes:
 * - updating auxiliary model parameters,
 * - constructing the search spaces,
 * - evaluating the log-joint probabilities,
 * - updating the gc sets,
 * - selecting the C_prime most likely components for each data point, and
 * - normalizing the variational distributions.
 *
 * Finally, the negative variational free energy (equivalently, the negative
 * evidence lower bound -- ELBO) averaged over all data points is returned.
 *
 * @param X Data matrix containing the datapoints.
 * @param model Model instance.
 * @param update_var_params If true, updates the Kn set and gc set.
 * @param beta Inverse temperature used for annealing.
 *
 * @return The average negative variational free energy per data point.
 */
template <class Model>
precision_t Variational::E_step(cRef<Matrix<>> X, Model& model, const bool update_var_params,
                                const precision_t& beta) {
    model.auxiliary();
    if (update_var_params and C_prime != C) {
        E_step_construct_Sn(model);
    }
    number_ljs = E_step_ljs(X, model);
    if (update_var_params and C_prime != C) {
        E_step_update_gc_set(model);
        E_step_select();
    }
    // TODO: model.objective_correction for beta unequal 1
    return -(E_step_normalize(beta) + model.objective_correction) / N;
}

/**
 * @brief Performs the M-step of the variational EM algorithm.
 *
 * The M-step updates the model parameters using the current variational
 * posterior distributions. If hard assignments are disabled, the current
 * partition of each component is constructed before the update.
 *
 * Model-specific parameter updates are performed independently for each active
 * component. After all components have been updated, the model finalizes the
 * parameter estimates. Optionally, discarded components are relocated to avoid
 * permanently inactive components.
 *
 * @param X Data matrix containing the data points.
 * @param model Model instance to be updated.
 */
template <class Model>
void Variational::M_step(cRef<Matrix<>> X, Model& model) {
    if (!M_step_hard) {
        get_partition();
    }

#pragma omp parallel
    {
        model.M_step_allocate([&](auto&... m_step_args) {
#pragma omp for schedule(dynamic, 1)
            for (size_t c = 0; c < C; c++) {
                if (!model.Mask[c]) continue;
                model.M_step_reset(c, m_step_args...);

                for (auto& _partition : partition) {
                    for (auto& [n, q_nc] : _partition[c]) {
                        q_nc = M_step_hard ? 1.0 : q_nc;
                        if (q_nc > 0.) {
                            model.M_step_accumulate(X.row(n), q_nc, c, m_step_args...);
                        }
                    }
                }
                model.M_step_update(c, m_step_args...);
            }
        });
    }
    model.M_step_finalize(N);
    if (relocate_discarded) {
        relocate_discarded_components(model);
    }
}

/**
 * @brief Relocates discarded model components.
 *
 * Components that became inactive during the EM-step are reassigned by
 * duplicating information from randomly selected active components. The
 * selection is performed according to the current component prior.
 *
 * During relocation, the gc set is updated to ensure that the
 * relocated components become reachable during subsequent search-space
 * expansion. Finally, the model-specific relocation procedure is invoked and
 * the component priors are adjusted.
 *
 * @param model Model instance whose inactive components are relocated.
 *
 * @note The number of relocated components is stored in `C_relocate`.
 */
template <class Model>
void Variational::relocate_discarded_components(Model& model) {
    C_relocate = C - model.C_active;
    if (C_relocate == 0) {  // nothing to relocate
        return;
    }

    // Discarded components can not be drawn, as they have zero prior.
    std::discrete_distribution<size_t> sample_from_prior(model.P.begin(), model.P.end());
    set_t<> set;
    size_t c_split = 0;

    relocate_from.clear();
    relocate_from.reserve(C_relocate);
    relocate_to.clear();
    relocate_to.reserve(C_relocate);

    set.reserve(G);

    for (size_t c = 0; c < C; c++) {
        if (model.Mask[c]) continue;
        c_split = sample_from_prior(rng());

        relocate_from.push_back(c);
        relocate_to.push_back(c_split);
        model.Mask[c] = true;

        for (const size_t g : gc_set[c_split]) {
            if ((g != c_split) and model.Mask[g]) {
                set.insert(g);
            }
        }
        if (set.count(c) == 0) {
            while (set.size() >= G - 1) {
                set.erase(set.cbegin());  // 'randomly' remove components
            }
            set.insert(c_split);
            set.insert(c);
            gc_set[c_split].clear();
            for (const size_t g : set) {
                gc_set[c_split].push_back(g);
            }
        }
        set.clear();
    }
    model.C_active = C;
    model.relocate_discarded_components(relocate_from, relocate_to, rng);
    model.P_adjust();
    return;
}

/**
 * @brief Performs one iteration of the variational EM algorithm.
 *
 * Executes a complete E-step and, if fit is True, subsequently performs the
 * M-step. The returned objective corresponds to the average negative
 * variational free energy after the E-step.
 *
 * @param X Data matrix containing the data points.
 * @param model Model instance.
 * @param fit If true, performs the M-step after the E-step.
 * @param update_var_params If true, updates the Kn set and gc set.
 * @param beta Inverse temperature used for annealing.
 *
 * @return The average negative variational free energy.
 */
template <class Model>
precision_t Variational::EM_step(cRef<Matrix<>> X, Model& model, const bool fit, const bool update_var_params,
                                 const precision_t& beta) {
    precision_t objective = E_step(X, model, update_var_params, beta);
    if (fit) {
        M_step(X, model);
    }
    return objective;
}

std::unordered_map<size_t, precision_t> Variational::q_map(size_t n) const {
    checkIndex(n, qs.size());
    std::unordered_map<size_t, precision_t> map;
    for (const auto& el : qs[n]) {
        map.insert(el);
    }
    return map;
}

void Variational::q_in(size_t n, const std::unordered_map<size_t, precision_t>& map) {
    checkIndex(n, N);
    if (map.size() != qs[n].size()) {
        throw std::invalid_argument("Invalid map.size()!\n");
    }
    for (const auto& it : map) {
        checkIndex(it.first, C);
    }
    qs[n].clear();
    qs[n].reserve(map.size());
    for (const auto& it : map) {
        qs[n].push_back(it);
    }
}

auto Variational::approx_map(size_t n) const {
    auto it = std::max_element(qs[n].begin(), qs[n].end(),
                               [](auto& lhs, auto& rhs) -> bool { return lhs.second < rhs.second; });
    return std::make_pair(it->first, it->second);
}

auto Variational::q_to_sparse_matrix(void) const {
    std::vector<Eigen::Triplet<precision_t>>
        coeff;  // each Triplet in this vector is a non-zero entry with (row index, column index, value)
    SparseMatrix<> sp_mat(N, C);
    coeff.reserve(qs[0].size() * qs.size());
    for (size_t n = 0; n < N; n++) {
        for (const auto& it : qs[n]) {
            coeff.emplace_back(n, it.first, it.second);
        }
    }
    sp_mat.setFromTriplets(coeff.begin(), coeff.end());
    return sp_mat;
}

auto Variational::q_shrinked_to_sparse_matrix(cRef<Vector<bool>> Mask) const {
    checkSize(Mask, C);
    size_t C_active = Mask.count();
    int i = 0;
    Vector<int> idx(C);
    Vector<size_t> rev_idx(C_active);
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            idx[c] = i;
            rev_idx[i] = c;
            i++;
        } else {
            idx[c] = -1;
        }
    }
    std::vector<Eigen::Triplet<precision_t>>
        coeff;  // each Triplet in this vector is a non-zero entry with (row index, column index, value)
    SparseMatrix<> sp_mat(N, C_active);
    coeff.reserve(qs[0].size() * qs.size());
    for (size_t n = 0; n < N; n++) {
        for (const auto& [c, value] : qs[n]) {
            if (Mask[c]) {
                coeff.emplace_back(n, idx[c], value);
            }
        }
    }

    sp_mat.setFromTriplets(coeff.begin(), coeff.end());
    return std::make_tuple(sp_mat, rev_idx);
}

void Variational::q_from_sparse_matrix(const SparseMatrix<> sp_mat) {
    checkSize(sp_mat, N, C);
    if ((size_t)sp_mat.nonZeros() != qs[0].size() * qs.size()) {
        std::stringstream msg;
        msg << "expected input having " << qs[0].size() * qs.size() << " non zero coefficients, but got "
            << sp_mat.nonZeros();
        throw std::invalid_argument(msg.str());
    }
    for (size_t n = 0; n < N; n++) {
        if ((size_t)sp_mat.row(n).nonZeros() != qs[n].size()) {
            std::stringstream msg;
            msg << "expected row " << n << " having " << qs[n].size() << " non zero coefficients, but got "
                << sp_mat.row(n).nonZeros();
            throw std::invalid_argument(msg.str());
        }
        qs[n].clear();
        for (SparseMatrix<>::InnerIterator it(sp_mat, n); it; ++it) {
            qs[n].emplace_back(it.col(), it.value());
        }
    }
}

auto Variational::get_gc_set(const size_t c) {
    checkIndex(c, C);
    return Eigen::Map<Vector<size_t>, Eigen::Unaligned>(gc_set[c].data(), gc_set[c].size());
}

Vector<size_t> Variational::indices() const {
    Vector<size_t> indices(N);
    for (size_t n = 0; n < N; n++) {
        indices(n) = std::max_element(qs[n].begin(), qs[n].end(), [](auto& lhs, auto& rhs) -> bool {
                         return lhs.second < rhs.second;
                     })->first;
    }
    return indices;
}

auto Variational::gc_set_to_sparse_matrix(void) const {
    // each Triplet in this vector is a non-zero entry with (row index, column index, value)
    std::vector<Eigen::Triplet<precision_t>> coeff;
    SparseMatrix<> sp_mat(C, C);
    coeff.reserve(gc_set[0].size() * gc_set.size());
    for (size_t c = 0; c < C; c++) {
        for (const size_t& k : gc_set[c]) {
            coeff.emplace_back(c, k, 1.0);
        }
    }
    sp_mat.setFromTriplets(coeff.begin(), coeff.end());
    return sp_mat;
}

void Variational::gc_set_from_sparse_matrix(const SparseMatrix<> sp_mat) {
    bool c_not_in;
    checkSize(sp_mat, C, C);
    if ((size_t)sp_mat.nonZeros() != gc_set[0].size() * gc_set.size()) {
        std::stringstream msg;
        msg << "expected input having " << gc_set[0].size() * gc_set.size()
            << " non zero coefficients, but got " << sp_mat.nonZeros();
        throw std::invalid_argument(msg.str());
    }
    for (size_t c = 0; c < C; c++) {
        if ((size_t)sp_mat.row(c).nonZeros() != gc_set[0].size()) {
            std::stringstream msg;
            msg << "expected row " << c << " having " << gc_set[0].size() << " non zero coefficients, but got "
                << sp_mat.row(c).nonZeros();
            throw std::invalid_argument(msg.str());
        }
        gc_set[c].clear();
        c_not_in = true;
        for (SparseMatrix<>::InnerIterator it(sp_mat, c); it; ++it) {
            gc_set[c].emplace_back(it.col());
            if ((size_t)it.col() == c) {
                c_not_in = false;
            }
        }
        if (c_not_in) {
            std::stringstream msg;
            msg << "row " << c << " is not containing itself";
            throw std::invalid_argument(msg.str());
        }
    }
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

template <typename... Model>
void Variational::bind(py::module_& m) {
    py::class_<Variational> Variational_class_(m, "Variational", py::module_local());

    /* Bindings specific to Variational */

    Variational_class_.def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, bool, bool, std::string>(),
                           "N"_a, "C"_a, "C_prime"_a, "G"_a, "E"_a, "seed"_a, "relocate_discarded"_a = true,
                           "hard"_a = false, "sim_measure"_a = "");
    Variational_class_.def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, cRef<Vector<size_t>>, bool,
                                    bool, std::string>(),
                           "N"_a, "C"_a, "C_prime"_a, "G"_a, "E"_a, "seed"_a, "indices"_a,
                           "relocate_discarded"_a = true, "hard"_a = false, "sim_measure"_a = "");
    // constructor with indices=None in python:
    Variational_class_.def(py::init([](size_t N, size_t C, size_t C_prime, size_t G, size_t E, size_t seed,
                                       py::none, bool relocate_discarded, bool hard, std::string sim_measure) {
                               return std::unique_ptr<Variational>(new Variational(
                                   N, C, C_prime, G, E, seed, relocate_discarded, hard, sim_measure));
                           }),
                           "N"_a, "C"_a, "C_prime"_a, "G"_a, "E"_a, "seed"_a, "indices"_a,
                           "relocate_discarded"_a = true, "hard"_a = false, "sim_measure"_a = "");
    Variational_class_.def_readonly("N", &Variational::N);
    Variational_class_.def_readonly("C", &Variational::C);
    Variational_class_.def_readonly("C_prime", &Variational::C_prime);
    Variational_class_.def_readonly("G", &Variational::G);
    Variational_class_.def_readwrite("E", &Variational::E);

    Variational_class_.def_property("q", &Variational::q_to_sparse_matrix, &Variational::q_from_sparse_matrix);
    Variational_class_.def_property("g", &Variational::gc_set_to_sparse_matrix,
                                    &Variational::gc_set_from_sparse_matrix);
    // gets converted to or from a scipy sparse csr matrix, creates a copy in both directions

    Variational_class_.def_readonly("number_ljs", &Variational::number_ljs);
    Variational_class_.def_readonly("initial_seed", &Variational::initial_seed);
    Variational_class_.def_readonly("C_relocate", &Variational::C_relocate);

    Variational_class_.def("q_map", &Variational::q_map, "n"_a);
    Variational_class_.def("q_in", &Variational::q_in, "n"_a, "map"_a);
    Variational_class_.def("gc_set", &Variational::get_gc_set, "c"_a);
    Variational_class_.def("approx_map", &Variational::approx_map, "n"_a);
    Variational_class_.def("indices", &Variational::indices);
    Variational_class_.def("q_shrinked", &Variational::q_shrinked_to_sparse_matrix, "mask"_a.noconvert());

    (Variational_class_.def(
         "_E_step",
         [](Variational& self, cRef<Matrix<>> X, Model& model, bool update_var_params,
            precision_t beta) -> auto { return self.E_step(X, model, update_var_params, beta); },
         "X"_a.noconvert(), "model"_a, "update_var_params"_a = true, "beta"_a = 1.0),
     ...);

    (Variational_class_.def(
         "_M_step",
         [](Variational& self, cRef<Matrix<>> X, Model& model) -> auto { return self.M_step(X, model); },
         "X"_a.noconvert(), "model"_a),
     ...);

    (Variational_class_.def(
         "_EM_step",
         [](Variational& self, cRef<Matrix<>> X, Model& model, bool fit, bool update_var_params,
            precision_t beta) -> auto { return self.EM_step(X, model, fit, update_var_params, beta); },
         "X"_a.noconvert(), "model"_a, "fit"_a = true, "update_var_params"_a = true, "beta"_a = 1.0),
     ...);
}
#endif
