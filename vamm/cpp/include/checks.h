/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once
#include <Eigen/Core>
#include <sstream>
#include <stdexcept>
#include <string>

template <typename Derived>
void checkSize(const Eigen::EigenBase<Derived>& in, long size) {
    if (in.size() != size) {
        std::stringstream msg;
        msg << "expected input of size " << size << ", but got size " << in.size();
        throw std::invalid_argument(msg.str());
    }
}

template <typename Derived>
void checkSize(const Eigen::EigenBase<Derived>& in, long rows, long cols) {
    if (in.rows() != rows or in.cols() != cols) {
        std::stringstream msg;
        msg << "expected input of shape (" << rows << "," << cols << "), but got shape (" << in.rows() << ","
            << in.cols() << ")";
        throw std::invalid_argument(msg.str());
    }
}

template <typename Derived, typename scalar>
void checkLow(const Eigen::MatrixBase<Derived>& in, scalar low, bool include_low = true) {
    std::stringstream msg;
    msg << "input values must be ";
    if (include_low) {
        if ((in.array() < low).any()) {
            msg << ">= " << low;
            throw std::invalid_argument(msg.str());
        }
    } else {
        if ((in.array() <= low).any()) {
            msg << "> " << low;
            throw std::invalid_argument(msg.str());
        }
    }
}

template <typename Derived, typename scalar>
void checkHigh(const Eigen::MatrixBase<Derived>& in, scalar high, bool include_high = true) {
    std::stringstream msg;
    msg << "input values must be ";
    if (include_high) {
        if ((in.array() > high).any()) {
            msg << "<= " << high;
            throw std::invalid_argument(msg.str());
        }
    } else {
        if ((in.array() >= high).any()) {
            msg << "< " << high;
            throw std::invalid_argument(msg.str());
        }
    }
}

template <typename Derived, typename scalar>
void checkRange(const Eigen::MatrixBase<Derived>& in, scalar low, scalar high, bool include_low = true,
                bool include_high = true) {
    std::stringstream msg;
    msg << "input values must be in ";
    bool throw_error = false;

    // check lower value
    if (include_low) {
        msg << "[" << low << ", ";
        if ((in.array() < low).any()) {
            throw_error = true;
        }
    } else {
        msg << "(" << low << ", ";
        if ((in.array() <= low).any()) {
            throw_error = true;
        }
    }
    // check upper value
    if (include_high) {
        msg << high << "]";
        if ((in.array() > high).any()) {
            throw_error = true;
        }
    } else {
        msg << high << ")";
        if ((in.array() >= high).any()) {
            throw_error = true;
        }
    }
    if (throw_error) {
        throw std::invalid_argument(msg.str());
    }
}

void checkIndex(size_t index, size_t size) {
    if (index >= size) {
        std::stringstream msg;
        msg << "index " << index << " is out of range [0, " << size - 1 << "]";
        throw std::out_of_range(msg.str());
    }
}