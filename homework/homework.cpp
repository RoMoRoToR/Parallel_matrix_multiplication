#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

// Вставьте функции blas_dgemm и blas_dgemm_sequential сюда
void blas_dgemm_sequential(int N, double* A, double* B, double* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i + k * N] * B[k + j * N];  // учтите column-major порядок
            }
            C[i + j * N] = sum;
        }
    }
}
void blas_dgemm(int N, double* A, double* B, double* C) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i + k * N] * B[k + j * N];
            }
            C[i + j * N] = sum;
        }
    }
}


int main() {
    int N = 500;
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    std::vector<double> C_check(N * N);  // для проверки корректности

    // Заполнение матриц случайными числами
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < N * N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // Проверка последовательной реализации
    auto start = std::chrono::high_resolution_clock::now();
    blas_dgemm_sequential(N, A.data(), B.data(), C_check.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Sequential Duration: " << duration.count() << "us" << std::endl;

    // Проверка параллельной реализации
    start = std::chrono::high_resolution_clock::now();
    blas_dgemm(N, A.data(), B.data(), C.data());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Parallel Duration: " << duration.count() << "us" << std::endl;

    // Проверка на равенство
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(C[i] - C_check[i]) > 1e-10) {
            std::cout << "Results are not equal!" << std::endl;
            return 1;
        }
    }
    std::cout << "Results are equal!" << std::endl;

    return 0;
}
