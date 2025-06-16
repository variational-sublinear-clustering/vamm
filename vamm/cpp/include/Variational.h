/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
#include <pybind11/functional.h>
#endif

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <list>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Numpy.h"
#include "SpVarEM.h"
#include "omp_helper.h"

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

class Variational : public SpVarEM<Variational> {
   public:
    size_t N;
    size_t C;
    size_t C_prime;
    size_t G;

    Vector<size_t> E;
    std::vector<std::vector<size_t>> graph;
    std::vector<std::vector<q_t>> all_list;
    size_t number_ljs;
    Random<std::mt19937_64> rng;

    size_t initial_seed;

    template <class Model>
    void E_step_expand(const Model& model);

    void get_partition(void);

    precision_t (*compute_relevance)(const precision_t&, const precision_t&);
    void (*finalize_sum_ljs)(std::vector<Triplet>&, cRef<Vector<>>, const size_t);

    template <class Model>
    void E_step_update_graph(const Model& model);

    void E_step_select(void);

    void init_shared_(const std::string&);

    Variational(size_t, size_t, size_t, size_t, size_t, size_t, bool, std::string);
    Variational(size_t, size_t, size_t, size_t, size_t, size_t, cRef<Vector<size_t>>, bool, std::string);

    template <class Model>
    precision_t E_step(cRef<Matrix<>> X, Model& model, const bool update_var_params, const precision_t& beta);

    size_t get_N() const;
    size_t get_C() const;
    size_t get_C_prime() const;
    size_t get_G() const;
    Vector<size_t>& get_E();

    size_t get_number_ljs() const;
    size_t get_initial_seed() const;

    void set_E(cRef<Vector<size_t>> E_);
    void fill_E(const size_t E_);
    void set_num_random(const size_t n, const size_t E_);

    auto group(const size_t c);

    Vector<size_t> indices(void) const;

    auto graph_to_sparse_matrix(void) const;
    void graph_from_sparse_matrix(const SparseMatrix<> sp_mat);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
    template <typename... Model>
    static void bind(py::module_& m);
#endif
};

//--------------------------------------------------------------------------------------------------------------------//

precision_t compute_KL_divergence(const precision_t& log_prob1, const precision_t& log_prob2) {
    return log_prob1 - log_prob2;
}
precision_t compute_distance(const precision_t&, const precision_t& log_prob) { return -log_prob; }

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

Variational::Variational(size_t _N, size_t _C, size_t _C_prime, size_t _G, size_t _E, size_t _seed,
                         bool _hard = false, std::string _g_approx = "") :
    SpVarEM(_N, _C, _hard),
    N(_N),
    C(_C),
    C_prime(_C_prime),
    G(_G),
    E(_N),
    graph(_C),
    all_list(get_max_threads()),
    number_ljs(0),
    rng(_seed),
    initial_seed(_seed) {
    E.fill(_E);
    init_shared_(_g_approx);
    if (C_prime != C) {
#pragma omp parallel
        {
            set_t<> set;
            std::uniform_int_distribution<size_t> rand_int(0, C - 1);

            set.reserve(C_prime);
#pragma omp for
            for (size_t n = 0; n < N; n++) {
                qs[n].reserve(C_prime * G + E[n]);
                while (set.size() < C_prime) {
                    set.insert(rand_int(rng()));
                }
                for (size_t c : set) {
                    qs[n].emplace_back(c, 1.);
                }
                set.clear();
            }
        }
    }
}

Variational::Variational(size_t _N, size_t _C, size_t _C_prime, size_t _G, size_t _E, size_t _seed,
                         cRef<Vector<size_t>> indices, bool _hard = false, std::string _g_approx = "") :
    SpVarEM(_N, _C, _hard),
    N(_N),
    C(_C),
    C_prime(_C_prime),
    G(_G),
    E(_N),
    graph(_C),
    all_list(get_max_threads()),
    number_ljs(0),
    rng(_seed),
    initial_seed(_seed) {
    size_t indices_size = indices.size();
    E.fill(_E);
    init_shared_(_g_approx);
    if ((indices_size != N) and (indices_size != C)) {
        throw std::invalid_argument("indices must be of size N or C");
    }
    if (C_prime != C) {
        if (indices_size == C) {
            for (size_t c = 0; c < C; c++) {
                qs[indices[c]].emplace_back(c, 1.);
            }
        }
#pragma omp parallel
        {
            set_t<> set;
            std::uniform_int_distribution<size_t> rand_int(0, C - 1);

            set.reserve(C_prime);
#pragma omp for
            for (size_t n = 0; n < N; n++) {
                qs[n].reserve(C_prime * G + E[n]);
                if (indices_size == N) {
                    set.insert(indices[n]);
                } else {
                    if (qs[n].size() == 1) {
                        set.insert(qs[n][0].first);
                        qs[n].pop_back();
                    }
                }
                while (set.size() < C_prime) {
                    set.insert(rand_int(rng()));
                }
                for (size_t c : set) {
                    qs[n].emplace_back(c, 1.);
                }
                set.clear();
            }
        }
    }
}

void Variational::init_shared_(const std::string& _g_approx = "") {
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

    if (_g_approx == "distance") {
        compute_relevance = compute_distance;
        finalize_sum_ljs = norm;
    } else {
        compute_relevance = compute_KL_divergence;
        finalize_sum_ljs = remove_prior_norm;
    }
#pragma omp parallel
    {
        set_t<> set;
        size_t thread_num = get_thread_num();
        std::uniform_int_distribution<size_t> rand_int(0, C - 1);

        if (C_prime == C) {
#pragma omp for
            for (size_t n = 0; n < N; n++) {
                qs[n].reserve(C_prime * G + E[n]);
                for (size_t c = 0; c < C; c++) {
                    qs[n].emplace_back(c, 1.);
                }
            }
        }
        set.reserve(G);
#pragma omp for
        for (size_t c = 0; c < C; c++) {
            graph[c].reserve(G);
            if (G == C) {
                for (size_t g = 0; g < G; g++) {
                    graph[c].emplace_back(g);
                }
            } else {
                set.insert(c);
                while (set.size() < G) {
                    set.insert(rand_int(rng()));
                }
                for (size_t g : set) {
                    graph[c].emplace_back(g);
                }
            }
            set.clear();
        }

        all_list[thread_num] = std::vector<q_t>(C);
    }
}

template <class Model>
void Variational::E_step_expand(const Model& model) {
#pragma omp parallel
    {
        size_t c;
        set_t<> set;
        std::uniform_int_distribution<size_t> rand_int(0, C - 1);

#pragma omp for
        for (size_t n = 0; n < N; n++) {
            for (const auto& [k, _] : qs[n]) {
                if (model.Mask[k]) {
                    for (const auto g : graph[k]) {
                        if (model.Mask[g]) {
                            set.insert(g);
                        }
                    }
                }
            }
            for (size_t e = 0; e < E[n]; e++) {
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

void Variational::get_partition(void) {
#pragma omp parallel
    {
        size_t end;
        size_t thread_num = get_thread_num();
        std::fill(all_list[thread_num].begin(), all_list[thread_num].end(), q_t(0));
#pragma omp for
        for (size_t n = 0; n < N; n++) {
            end = std::min(C_prime, qs[n].size());
            // TODO: move partial sorting to another part in the code?
            if (qs[n].size() > C_prime) {
                std::nth_element(qs[n].begin(), qs[n].begin() + C_prime, qs[n].end(),
                                 [](auto& lhs, auto& rhs) -> bool { return lhs.second > rhs.second; });
            }
            auto it = std::max_element(qs[n].begin(), qs[n].begin() + end,
                                       [](auto& lhs, auto& rhs) -> bool { return lhs.second < rhs.second; });
            all_list[thread_num][it->first].emplace_back(n, it->second);
        }
        for (size_t c = 0; c < C; c++) {
            all_list[thread_num][c].shrink_to_fit();
        }
    }
}

template <class Model>
void Variational::E_step_update_graph(const Model& model) {
    if (G == 1) {
        return;  // graph[c] constains only c, no update needed
    }
    /* */
    get_partition();
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
            if (model.Mask[c]) {
                graph[c].clear();
                graph[c].push_back(c);
                for (const auto& list : all_list) {
                    if (!list[c].empty()) {
                        for (const auto& [n, log_prob_c] : list[c]) {
                            for (const auto& [k, log_prob] : qs[n]) { /* contains search spaces */
                                if (k != c) {
                                    auto el = index_map.find(k);
                                    if (el != index_map.end()) {
                                        candidates[el->second].num_ljs++;
                                        candidates[el->second].sum_ljs +=
                                            compute_relevance(log_prob_c, log_prob);
                                    } else {
                                        index_map.emplace(k, candidates.size());
                                        candidates.emplace_back(k, 1, compute_relevance(log_prob_c, log_prob));
                                    }
                                }
                            }
                        }
                    }
                }
                if (!index_map.empty()) {
                    finalize_sum_ljs(candidates, model.P_log, c);
                    if (index_map.size() >= G) {
                        std::nth_element(
                            candidates.begin(), candidates.begin() + (G - 1), candidates.end(),
                            [](auto& lhs, auto& rhs) -> bool { return lhs.sum_ljs < rhs.sum_ljs; });
                    }
                    end = std::min(G - 1, index_map.size());
                    for (auto it = candidates.begin(); it != candidates.begin() + end; ++it) {
                        graph[c].push_back(it->c);
                    }
                }
                candidates.clear();
                index_map.clear();
            }
        }
    }
    return;
}

void Variational::E_step_select(void) {
#pragma omp parallel
    {
        if (C_prime == 1) {
            std::pair<size_t, precision_t> c;
#pragma omp for
            for (size_t n = 0; n < N; n++) {
                auto it = std::max_element(qs[n].begin(), qs[n].end(), [](auto& lhs, auto& rhs) -> bool {
                    return lhs.second < rhs.second;
                });
                c = {it->first, it->second};  // make a copy before clearing q(n);
                qs[n].clear();
                qs[n].push_back(c);
            }
        } else {
#pragma omp for
            for (size_t n = 0; n < N; n++) {
                if (C_prime < qs[n].size()) {
                    qs[n].resize(C_prime);
                }
            }
        }
    }
}

template <class Model>
precision_t Variational::E_step(cRef<Matrix<>> X, Model& model, const bool update_var_params,
                                const precision_t& beta) {
    model.auxiliary();
    if (update_var_params and C_prime != C) {
        E_step_expand(model);
    }
    number_ljs = E_step_ljs(X, model);
    if (update_var_params and C_prime != C) {
        E_step_update_graph(model);
        E_step_select();
    }
    // TODO: model.objective_correction for beta unequal 1
    return -(E_step_normalize(beta) + model.objective_correction) / N;
}

size_t Variational::get_N() const { return N; }

size_t Variational::get_C() const { return C; }

size_t Variational::get_C_prime() const { return C_prime; }

size_t Variational::get_G() const { return G; }

Vector<size_t>& Variational::get_E(void) { return E; }

size_t Variational::get_number_ljs() const { return number_ljs; } /* Fix with number of active components */

size_t Variational::get_initial_seed() const { return initial_seed; }

void Variational::set_E(cRef<Vector<size_t>> E_) {
    checkSize(E_, N);
    E = E_;
}

void Variational::fill_E(size_t E_) { E.fill(E_); }

void Variational::set_num_random(const size_t n, const size_t E_) {
    checkIndex(n, N);
    E[n] = E_;
}

auto Variational::group(const size_t c) {
    checkIndex(c, C);
    return Eigen::Map<Vector<size_t>, Eigen::Unaligned>(graph[c].data(), graph[c].size());
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

auto Variational::graph_to_sparse_matrix(void) const {
    // each Triplet in this vector is a non-zero entry with (row index, column index, value)
    std::vector<Eigen::Triplet<precision_t>> coeff;
    SparseMatrix<> sp_mat(C, C);
    coeff.reserve(graph[0].size() * graph.size());
    for (size_t c = 0; c < C; c++) {
        for (const size_t& k : graph[c]) {
            coeff.emplace_back(c, k, 1.0);
        }
    }
    sp_mat.setFromTriplets(coeff.begin(), coeff.end());
    return sp_mat;
}

void Variational::graph_from_sparse_matrix(const SparseMatrix<> sp_mat) {
    bool c_not_in;
    checkSize(sp_mat, C, C);
    if ((size_t)sp_mat.nonZeros() != graph[0].size() * graph.size()) {
        std::stringstream msg;
        msg << "expected input having " << graph[0].size() * graph.size() << " non zero coefficients, but got "
            << sp_mat.nonZeros();
        throw std::invalid_argument(msg.str());
    }
    for (size_t c = 0; c < C; c++) {
        if ((size_t)sp_mat.row(c).nonZeros() != graph[0].size()) {
            std::stringstream msg;
            msg << "expected row " << c << " having " << graph[0].size() << " non zero coefficients, but got "
                << sp_mat.row(c).nonZeros();
            throw std::invalid_argument(msg.str());
        }
        graph[c].clear();
        c_not_in = true;
        for (SparseMatrix<>::InnerIterator it(sp_mat, c); it; ++it) {
            graph[c].emplace_back(it.col());
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
    py::class_<Variational> Variational_class_(m, "Variational", py::module_local(), R"(
    Expectation Maximization for mixture models with truncated posterior as variational distributions.

    Parameters
    ----------
    N : int
        Number of data points of the dataset the algorithm will be trained with.
    C : int
        Number of components.
    C_prime : int
        Number of non-zero elements in truncated posterior.
    G : int
        Component neighborhood size. Each component keeps track of G-1 closes neighbors.
    E : int
        Number of randomly added components.
    H : int
        Dimensionality of the factors.
    seed : int
        The seed of the random number generator.
    indices : np.ndarray or None, optional
        Indices of data points uses as initial component centers. Used for initializing the K-Sets and component
        neighborhood. Defaults to None.
    hard : bool, optional
        Whether to use hard assignment in the M-step. Defaults to False.
    )");

    /* Bindings specific to Variational */

    Variational_class_.def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, bool, std::string>(),
                           "N"_a, "C"_a, "C_prime"_a, "G"_a, "E"_a, "seed"_a, "hard"_a = false,
                           "g_approx"_a = "");
    Variational_class_.def(
        py::init<size_t, size_t, size_t, size_t, size_t, size_t, cRef<Vector<size_t>>, bool, std::string>(),
        "N"_a, "C"_a, "C_prime"_a, "G"_a, "E"_a, "seed"_a, "indices"_a, "hard"_a = false, "g_approx"_a = "");
    // constructor with indices=None in python:
    Variational_class_.def(
        py::init([](size_t N, size_t C, size_t C_prime, size_t G, size_t E, size_t seed, py::none, bool hard,
                    std::string g_approx) {
            return std::unique_ptr<Variational>(new Variational(N, C, C_prime, G, E, seed, hard, g_approx));
        }),
        "N"_a, "C"_a, "C_prime"_a, "G"_a, "E"_a, "seed"_a, "indices"_a, "hard"_a = false, "g_approx"_a = "");
    Variational_class_.def_property_readonly("N", &Variational::get_N, "The number of data points.");
    Variational_class_.def_property_readonly("C", &Variational::get_C, "The number of components.");
    Variational_class_.def_property_readonly("C_prime", &Variational::get_C_prime,
                                             "The number of non-zero elements in truncated posterior.");
    Variational_class_.def_property_readonly("G", &Variational::get_G, "The component neighborhood size.");
    Variational_class_.def_property_readonly("number_ljs", &Variational::get_number_ljs,
                                             "The number of log-joint evaluations in the last E-Step.");
    Variational_class_.def_property_readonly("initial_seed", &Variational::get_initial_seed,
                                             "The inital seed for the random number generator.");
    Variational_class_.def_property("E", &Variational::get_E, &Variational::set_E,
                                    "The number of randomly added components for each data point. "
                                    "The index of the array corresponds to the index of the data point.");

    Variational_class_.def("fill_E", &Variational::fill_E, "E"_a, R"(
    Set the number of randomly added components for all data points.

    Parameters
    ----------
    E : int
        Number of randomly added components.

    Returns
    -------
    None
    )");

    Variational_class_.def("set_num_random", &Variational::set_num_random, "n"_a, "E"_a, R"(
    Set the number of randomly added components for the data point with index `n`.

    Parameters
    ----------
    n : int
        Index of the data point.
    E : int
        Number of randomly added components.

    Returns
    -------
    None
    )");

    Variational_class_.def("q_map", &Variational::q_map, "n"_a, R"(
    The variational distribution for the data point with index `n`.

    This method returns a dictionary with the C_prime non-zero elements in the variational distribution
    of the data point with index `n`. Keys are component indices and values are corresponding values
    of the variational distribution.

    Parameters
    ----------
    n : int
        Index of the data point.

    Returns
    -------
    Dict[int, float]
        Dictionary representing the variational distribution.
    )");

    Variational_class_.def("q_in", &Variational::q_in, "n"_a, "map"_a, R"(
    Set the variational distribution for the data point with index `n`.

    This method accepts a dictionary with C_prime elements that contain the variational distribution
    of the data point with index `n`. Keys are component indices and values are corresponding values
    of the variational distribution.

    Parameters
    ----------
    n : int
        Index of the data point.
    map : Dict[int, float]
        Dictionary representing the variational distribution.

    Returns
    -------
    None
    )");

    Variational_class_.def("group", &Variational::group, "c"_a, R"(
    Neighborhood of component `c`.

    This method retrieves the approximate neighborhood of the component with index `c`.
    The neighborhood always includes the index `c` itself and the G-1 indices of the
    nearest components found so far.

    Parameters
    ----------
    c : int
        Component index.

    Returns
    -------
    np.array[int]
        An array containing the indices of the component neighborhood.
    )");

    Variational_class_.def("approx_map", &Variational::approx_map, "n"_a, R"(
    Finds the index of the component with the larges variational distribution for the data 
    point with index `n`.

    This method finds the index of the component with the larges value of the variational distribution
    for the given data point within the k non-zero elements.

    Parameters
    ----------
    n : int
        Index of the data point.

    Returns
    -------
    index: int
        The index of the component.
    value: float
        The value of the variational distribution
    )");

    Variational_class_.def("indices", &Variational::indices, R"(
    Gets the indices of the components with the larges variational distribution for all data 
    point.

    This method finds the indices of the components with the larges value of the variational distribution 
    for all data points. The index of the resulting array corresponds to the
    index of the data point.

    Returns
    -------
    indices: npt.ndarray
        An array containing the indices of the components.
    )");
    Variational_class_.def_property("g", &Variational::graph_to_sparse_matrix,
                                    &Variational::graph_from_sparse_matrix,
                                    "The neighborhood graph to or from a scipy sparse csr matrix. "
                                    "This requires copies in both directions.");

    /* From base class */

    bind_q_sparse_matrix_interface(Variational_class_);

    bind_E_step<precision_t, Model...>(Variational_class_);
    bind_M_step<precision_t, Model...>(Variational_class_);
    bind_EM_step<precision_t, Model...>(Variational_class_);
}
#endif
