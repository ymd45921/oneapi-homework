#ifndef OneAPI_Homework_my_mat_hpp
#define OneAPI_Homework_my_mat_hpp
#pragma once

#include "my.hpp"

namespace my {

    // 使用上面的随机数生成器生成 axb 的随机矩阵，a 和 b 的大小由模板参数指定
    template <typename T, int a, int b>
    void random_matrix(T (&matrix)[a][b], rand<T> &rand) {
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++)
                matrix[i][j] = rand();
        }
    }

    // Host 暴力运算矩阵乘法
    template <typename T, int A, int B, int C>
    void matrix_multiply(const T (&a)[A][B], const T (&b)[B][C], T (&c)[A][C]) {
        for (int i = 0; i < A; i++) {
            for (int j = 0; j < C; j++) {
                T sum = 0;
                for (int k = 0; k < B; k++)
                    sum += a[i][k] * b[k][j];
                c[i][j] = sum;
            }
        }
    }

    // Host 暴力运算矩阵减法
    template <typename T, int A, int B>
    void matrix_subtract(const T (&a)[A][B], const T (&b)[A][B], T (&c)[A][B]) {
        for (int i = 0; i < A; i++) {
            for (int j = 0; j < B; j++)
                c[i][j] = a[i][j] - b[i][j];
        }
    }

    // 使用上面的浮点数 equal 函数检查两个矩阵是否相等
    template <typename T, int A, int B>
    bool check_matrix_equal(const T (&a)[A][B], const T (&b)[A][B], equal<T> &eq) {
        for (int i = 0; i < A; i++) {
            for (int j = 0; j < B; j++)
                if (!eq(a[i][j], b[i][j]))
                    return false;
        }
        return true;
    }

    // 将矩阵输出到文件
    template <typename T, int A, int B>
    void print_matrix(std::ostream &os, const T (&matrix)[A][B]) {
        for (int i = 0; i < A; i++) {
            for (int j = 0; j < B; j++)
                os << matrix[i][j] << " ";
            os << "\n";
        }
    }

    // 二维矩阵 RAII 封装
    template <typename T, int M, int N>
    class mat {
        T (*_data)[N];

    public:
        mat() {
            _data = new T[M][N];
        }

        mat(const T (&matrix)[M][N]) {
            _data = new T[M][N];
            std::memcpy(_data, matrix, sizeof(T) * M * N);
        }

        mat(const mat &other) {
            _data = new T[M][N];
            std::memcpy(_data, other._data, sizeof(T) * M * N);
        }

        mat(mat &&other) {
            _data = other._data;
            other._data = nullptr;
        }

        mat &operator=(const mat &other) {
            if (this != &other) {
                if (_data == nullptr)
                    _data = new T[M][N];
                std::memcpy(_data, other._data, sizeof(T) * M * N);
            }
            return *this;
        }

        mat &operator=(mat &&other) {
            if (this != &other) {
                if (_data != nullptr)
                    delete[] _data;
                _data = other._data;
                other._data = nullptr;
            }
            return *this;
        }

        bool operator==(const mat &other) const {
            return std::memcmp(_data, other._data, sizeof(T) * M * N) == 0;
        }

        bool operator!=(const mat &other) const {
            return std::memcmp(_data, other._data, sizeof(T) * M * N) != 0;
        }

        bool equal(const mat &other, equal<T> &eq) const {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    if (!eq(_data[i][j], other[i][j]))
                        return false;
            return true;
        }

        mat operator+(const mat &other) const {
            mat result;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    result[i][j] = _data[i][j] + other[i][j];
            return result;
        }

        mat operator-(const mat &other) const {
            mat result;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    result[i][j] = _data[i][j] - other[i][j];
            return result;
        }

        template <int P>
        mat<T, M, P> operator*(const mat<T, N, P> &other) const {
            mat<T, M, P> result;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < P; j++) {
                    T sum = 0;
                    for (int k = 0; k < N; k++)
                        sum += _data[i][k] * other[k][j];
                    result[i][j] = sum;
                }
            }
            return result;
        }

        std::ostream &operator<<(std::ostream &os) const {
            os << "my::mat(" << M << ", " << N << ") [\n";
            for (int i = 0; i < M; i++) {
                os << "  ";
                for (int j = 0; j < N; j++)
                    os << _data[i][j] << " ";
                os << "\n";
            }
            os << "]";
            return os;
        }

        T operator()(int i, int j) const {
            return _data[i][j];
        }

        auto get_offset() const {
            return [=](int i, int j) { return i * N + j; };
        }

        ~mat() {
            if (_data != nullptr)
                delete[] _data;
        }

        T *operator[](int i) {
            return _data[i];
        }

        const T *operator[](int i) const {
            return _data[i];
        }

        void random(rand<T> &rand) {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    _data[i][j] = rand();
        }

        void random(T min = 0, T max = 1) {
            rand<T> rand(min, max);
            random(rand);
        }

        T *begin() {
            return _data[0];
        }

        T *end() {
            return _data[0] + M * N;
        }

        const T *begin() const {
            return _data[0];
        }

        const T *end() const {
            return _data[0] + M * N;
        }

        const T *cbegin() const {
            return _data[0];
        }

        const T *cend() const {
            return _data[0] + M * N;
        }

        T *data() {
            return _data[0];
        }

        int size() const {
            return M * N;
        }

        std::size_t size_of() const {
            return sizeof(T) * M * N;
        }

        int rows() const {
            return M;
        }

        int cols() const {
            return N;
        }

        void fill(T value) {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    _data[i][j] = value;
        }

        void memset(int value) {
            std::memset(_data, value, sizeof(T) * M * N);
        }

        std::ostream &print_to(std::ostream &os) const {
            os << "my::mat(" << M << ", " << N << ") [\n";
            for (int i = 0; i < M; i++) {
                os << "  ";
                for (int j = 0; j < N; j++)
                    os << _data[i][j] << " ";
                os << "\n";
            }
            return os << "]";
        }

        mat<T, N, M> operator!() const {    // transpose
            mat<T, N, M> result;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    result[j][i] = _data[i][j];
            return result;
        }
    };

    template <typename T, int M, int N>
    std::ostream &operator<<(std::ostream &os, const mat<float, M, N> &matrix) {
        return matrix.operator<<(os);
    }
}

#endif /* OneAPI_Homework_my_mat_hpp */