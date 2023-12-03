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



}

#endif /* OneAPI_Homework_my_mat_hpp */