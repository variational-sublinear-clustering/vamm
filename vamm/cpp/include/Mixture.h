/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once
#include <Eigen/Dense>

#include "Numpy.h"
#include "Variational.h"
#include "checks.h"

template <typename Model>
class Mixture {
   public:
    size_t C;
    size_t C_active;
    size_t D;
    precision_t objective_correction;
    bool flat_prior;
    bool verbose_discard;
    Vector<bool> Mask;
    Vector<> P;
    Matrix<> M;
    Vector<> P_log;

    static constexpr bool loop_order_n = true;
    static constexpr precision_t eps_ = std::numeric_limits<precision_t>::min();

    bool valid(size_t c) const;

    bool checkFinite(cRef<Matrix<>> in) const;

    void discard(const size_t c);
    void discard(const size_t c, const std::string& msg);
    void P_adjust();

    precision_t precompute(cRef<Matrix<>> X) {
        // this method is used to save the value of precompute_() on the variable 'objective_correction'
        objective_correction = precompute_(X);
        return objective_correction;
    }

    virtual void relocate_discarded_components(std::vector<size_t>&, std::vector<size_t>&,
                                               Random<std::mt19937_64>&) {
        throw std::runtime_error("relocate_discarded_components() not implemented for this model!");
    }

    virtual precision_t precompute_(cRef<Matrix<>>) { return 0.; }

    virtual void auxiliary_(const size_t) {}

    virtual void auxiliary() {
#pragma omp parallel for
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                auxiliary_(c);
            }
        }
    }

    size_t first_active() const;

    auto logProbMaxIndex_(cRef<Vector<>> x) const;

    precision_t log_joint_(cRef<Vector<>> x, size_t c) const;

    precision_t log_joint(cRef<Vector<>> x, size_t c);

    size_t map(cRef<Vector<>> x);

    auto map_k(cRef<Vector<>> x, size_t k);

    precision_t log_likelihood(cRef<Matrix<>> X);

    precision_t nll(cRef<Matrix<>> X);

    Vector<> log_prob(cRef<Matrix<>> X, cRef<Matrix<size_t>> indices);

   public:
    Mixture(size_t C_, size_t D_, bool flat_prior_) :
        C(C_),
        C_active(C_),
        D(D_),
        objective_correction(0.0),
        flat_prior(flat_prior_),
        verbose_discard(false),
        Mask(Vector<bool>::Ones(C_)),  // Ones is true for bool
        P(Vector<>::Ones(C_)),
        M(Matrix<>::Zero(C_, D_)),
        P_log(Vector<>::Ones(C_)) {
        if (C <= 0) {
            throw std::invalid_argument("C must be > 0");
        }
        if (D <= 0) {
            throw std::invalid_argument("D must be > 0");
        }
        P_adjust();
    }

    void set_P(cRef<Vector<>> P_);

    void set_M(cRef<Matrix<>>);

    Matrix<>& get_M();

    Vector<>& get_P(void);

    Vector<>& get_P_log(void);

    void set_Mask(cRef<Vector<bool>> Mask_);

    const Vector<bool> get_Mask(void);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
    template <typename T>
    static void bind_base(pybind11::class_<Model>& Model_class_);
#endif
};

template <typename Model>
void Mixture<Model>::set_P(cRef<Vector<>> P_) {
    if (flat_prior) {
        throw std::invalid_argument("Flat Prior can not be changed!");
    }
    checkSize(P_, C);
    checkLow(P_, 0., true);
    P = P_;
    for (size_t c = 0; c < C; c++) {
        if (P[c] == 0.) {
            discard(c, "prior is set to zero");
        }
    }
    P_adjust();
}

template <typename Model>
void Mixture<Model>::set_M(cRef<Matrix<>> M_) {
    checkSize(M_, C, D);
    M = M_;
}

template <typename Model>
Vector<>& Mixture<Model>::get_P(void) {
    return P;
}

template <typename Model>
Matrix<>& Mixture<Model>::get_M() {
    return M;
}

template <typename Model>
Vector<>& Mixture<Model>::get_P_log(void) {
    return P_log;
}

template <typename Model>
void Mixture<Model>::set_Mask(cRef<Vector<bool>> Mask_) {
    checkSize(Mask_, C);
    Mask = Mask_;
    C_active = Mask.count();
    for (size_t c = 0; c < C; c++) {
        if (!Mask[c]) {
            P[c] = 0.;
            P_log[c] = std::numeric_limits<precision_t>::quiet_NaN();
        }
    }
}

template <typename Model>
const Vector<bool> Mixture<Model>::get_Mask(void) {
    return Mask;
}

template <typename Model>
bool Mixture<Model>::valid(size_t c) const {
    return (c < C) && Mask[c];
}

template <typename Model>
bool Mixture<Model>::checkFinite(cRef<Matrix<>> in) const {
    return in.array().isFinite().all();
}

template <typename Model>
size_t Mixture<Model>::first_active() const {
    // find index of first active component
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            return c;
        }
    }
    return 0;
}

template <typename Model>
void Mixture<Model>::discard(const size_t c) {
    if (C_active == 1) {
        throw std::runtime_error("Refuse to remove last component!");
    }
    if (Mask[c]) {
        Mask[c] = false;
#pragma omp atomic
        C_active--;
        P[c] = 0.;
        P_log[c] = std::numeric_limits<precision_t>::quiet_NaN();
    }
}

template <typename Model>
void Mixture<Model>::discard(const size_t c, const std::string& msg) {
    discard(c);
    if (verbose_discard) {
        if (msg != "") {
            std::cerr << "discard component " << c << ": " << msg << "\n";
        } else {
            std::cerr << "discard component " << c << "\n";
        }
    }
}

template <typename Model>
void Mixture<Model>::P_adjust() {
    if (flat_prior) {
        precision_t flat_P = 1. / C_active;
        precision_t flat_P_log = std::log(flat_P);
#pragma omp parallel for
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                P[c] = flat_P;
                P_log[c] = flat_P_log;
            }
        }
    } else {
        precision_t sum = P.sum();
        P /= sum;
        if (!checkFinite(P)) {
            throw std::runtime_error("prior is not finite!");
        }
#pragma omp parallel for
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                if (P[c] <= 0) {
                    discard(c, "prior not positive");
                    continue;
                }
                P_log[c] = std::log(P[c]);
                if (!std::isfinite(P_log[c])) {
                    throw std::runtime_error("log of prior is not finite!");
                }
            }
        }
    }
}

// without correction for internal use
template <typename Model>
auto Mixture<Model>::logProbMaxIndex_(cRef<Vector<>> x) const {
    Vector<> LogJoints(C);
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            LogJoints[c] = static_cast<const Model*>(this)->log_joint_(x, c);
        } else {
            LogJoints[c] = -std::numeric_limits<precision_t>::infinity();
        }
    }
    size_t maxIndex;
    precision_t maxLogJoint = LogJoints.maxCoeff(&maxIndex);
    precision_t ret = std::log((LogJoints.array() - maxLogJoint).exp().sum()) + maxLogJoint;
    return std::make_pair(ret, maxIndex);
}

// log joint without correction for internal use
template <typename Model>
precision_t Mixture<Model>::log_joint_(cRef<Vector<>> x, size_t c) const {
    precision_t log_prob;
    static_cast<const Model*>(this)->E_step_allocate([&](auto&... e_step_args) -> auto {
        static_cast<const Model*>(this)->E_step_log_joint(x, c, log_prob, e_step_args...);
    });
    return log_prob;
}

// log joint with correction
template <typename Model>
precision_t Mixture<Model>::log_joint(cRef<Vector<>> x, size_t c) {
    precision_t correction = precompute_(x);
    auxiliary_(c);
    precision_t log_prob = log_joint_(x, c);
    return log_prob + correction;
}

template <typename Model>
size_t Mixture<Model>::map(cRef<Vector<>> x) {
    // precompute_(x) is not needed here since we only return the argmax index
    auxiliary();
    auto [log_prob, index] = logProbMaxIndex_(x);
    return index;
}

template <typename Model>
auto Mixture<Model>::map_k(cRef<Vector<>> x, size_t k) {
    if ((k > C_active) || (k == 0)) {
        throw std::invalid_argument("Invalid argument k!");
    }
    std::vector<std::pair<size_t, precision_t>> log_joints(C);
    precision_t correction = precompute_(x);
    auxiliary();

#pragma omp parallel for
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            precision_t log_prob = static_cast<const Model*>(this)->log_joint_(x, c);

            log_joints[c] = {c, log_prob};
        } else {
            log_joints[c] = {c, -std::numeric_limits<precision_t>::infinity()};
        }
    }
    std::nth_element(log_joints.begin(), log_joints.begin() + k, log_joints.end(),
                     [](auto lhs, auto rhs) -> bool { return lhs.second > rhs.second; });
    log_joints.resize(k);
    std::unordered_map<size_t, precision_t> map;
    for (auto& el : log_joints) {
        map.emplace(el.first, el.second + correction);
    }
    return map;
}

template <typename Model>
precision_t Mixture<Model>::log_likelihood(cRef<Matrix<>> X) {
    /* Not the most efficient implementation, but good enough for now */
    precision_t ll = 0;
    precision_t correction = precompute_(X);
    auxiliary();

#pragma omp parallel for
    for (size_t n = 0; n < (size_t)X.rows(); n++) {
        auto [log_prob, index] = logProbMaxIndex_(X.row(n));
        ll += log_prob;
    }
    return ll + correction;
}

template <typename Model>
precision_t Mixture<Model>::nll(cRef<Matrix<>> X) {
    return -log_likelihood(X) / X.rows();
}

template <typename Model>
Vector<> Mixture<Model>::log_prob(cRef<Matrix<>> X, cRef<Matrix<size_t>> indices) {
    size_t N = X.rows();
    size_t C_prime = indices.cols();
    Vector<> log_probabilities(N);

#pragma omp parallel
    {
        precision_t lim;
        precision_t sum;
        precision_t log_prob_;
        q_t q;
        q.reserve(C_prime);

        static_cast<const Model*>(this)->E_step_allocate([&](auto&... e_step_args) -> auto {
#pragma omp for  // schedule(dynamic,1)
            for (size_t n = 0; n < N; n++) {
                q.clear();
                for (auto c : indices.row(n)) {
                    static_cast<const Model*>(this)->E_step_log_joint(X.row(n), c, log_prob_, e_step_args...);
                    q.emplace_back(c, log_prob_);
                }
                normalize(q, 1.0, lim, sum);
                log_probabilities(n) = std::log(sum) + lim;
            }
        });
    }
    return log_probabilities;
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

template <typename Model>
template <typename T>
void Mixture<Model>::bind_base(pybind11::class_<Model>& Model_class_) {
    Model_class_.def_readonly("C", &Mixture<Model>::C);
    Model_class_.def_readonly("D", &Mixture<Model>::D);
    Model_class_.def_property("mask", &Mixture<Model>::get_Mask, &Mixture<Model>::set_Mask);
    Model_class_.def_property("prior", &Mixture<Model>::get_P, &Mixture<Model>::set_P);
    Model_class_.def_property("means", &Mixture<Model>::get_M, &Mixture<Model>::set_M);
    Model_class_.def_readwrite("flat_prior", &Mixture<Model>::flat_prior);
    Model_class_.def_readonly("active", &Mixture<Model>::C_active);
    Model_class_.def_property_readonly("log_prior", &Mixture<Model>::get_P_log);
    Model_class_.def_property_readonly("dtype",
                                       [](Model&) -> py::dtype { return py::dtype::of<precision_t>(); });
    Model_class_.def_readwrite("verbose_discard", &Mixture<Model>::verbose_discard);

    Model_class_.def("_precompute", &Mixture<Model>::precompute, "X"_a.noconvert());
    Model_class_.def(
        "discard",
        [](Model& self, size_t c) -> void {
            self.discard(c);
            self.P_adjust();
        },
        "c"_a);
    Model_class_.def("log_joint", &Mixture<Model>::log_joint, "x"_a.noconvert(), "c"_a);
    Model_class_.def("map", &Mixture<Model>::map, "x"_a.noconvert());
    Model_class_.def("map_k", &Mixture<Model>::map_k, py::return_value_policy::reference_internal,
                     "x"_a.noconvert(), "k"_a);
    Model_class_.def("ll", &Mixture<Model>::log_likelihood, "X"_a.noconvert());
    Model_class_.def("nll", &Mixture<Model>::nll, "X"_a.noconvert());
    Model_class_.def("log_prob", &Mixture<Model>::log_prob, "X"_a.noconvert(), "indices"_a.noconvert());
}

#endif
//--------------------------------------------------------------------------------------------------------------------//
