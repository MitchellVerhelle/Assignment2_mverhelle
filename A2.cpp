// This code structure is inspired by Professor (Roger) Lee's Numerical Methods python code, homeworks 1 & 2.
#include<cmath>
#include<algorithm>
#include<vector>
#include<chrono>
#include<iostream>

#include<array>
#include<memory>
#include<thread>
#include<cstddef>
#include<cstdint>

// PART 1

namespace part1 {

template<typename U>
struct Option {
    Option(U t, U k): T{t}, K{k} {}
    U T; // T = 1
    U K; // K = 100
};

template<typename U>
struct CallOption : Option<U> {
    CallOption(U t, U k): Option<U>(t, k) {};
};

template<typename U>
struct PutOption : Option<U> {
    PutOption(U t, U k): Option<U>(t, k) {};
};

template<typename U>
struct Dynamics {
    Dynamics() : S(U(0)), r(U(0)), v(U(0)) {}
    Dynamics(U S, U r, U v) : S(S), r(r), v(v) {}
    U S; // S = 90, 95, 100, 105, 110
    U r; // r = 0.03
    U v; // v = 0.3
};

template<typename U>
struct AnalyticsEngine {
    AnalyticsEngine(int N): N(N) {}
    inline U
    priceEuroCall_JarrowRudd(CallOption<U> const& o, Dynamics<U> const& dynamics)
    {
        U dt = o.T / N;
        U base = (dynamics.r - 0.5 * (dynamics.v * dynamics.v)) * dt;
        U diff = dynamics.v * std::sqrt(dt);
        U u = std::exp(base + diff);
        U d = std::exp(base - diff);
        U half_discount = 0.5*std::exp(-dynamics.r*dt);
        // initialize prices array at time T
        #pragma omd vectorize for
        std::vector<U> prices(N+1);
        for (int i = 0; i <= N; ++i) {
            prices[i] = std::max(static_cast<U>(dynamics.S * std::pow(u, i) * std::pow(d, N-i) - o.K), static_cast<U>(0.0));
        }
        // iterate backwards T time steps
        #pragma omd vectorize for
        for (int t = N-1; t >= 0; --t) {
            #pragma omd parallel for
            for (int i = 0; i <= t; ++i) {
                prices[i] = half_discount * (prices[i+1] + prices[i]);
            }
        }
        return prices[0];
    }
    inline U
    priceEuroPut_JarrowRudd(PutOption<U> const& o, Dynamics<U> const& dynamics)
    {
        U dt = o.T / N;
        U base = (dynamics.r - 0.5 * (dynamics.v * dynamics.v)) * dt;
        U diff = dynamics.v * std::sqrt(dt);
        U u = std::exp(base + diff);
        U d = std::exp(base - diff);
        U half_discount = 0.5*std::exp(-dynamics.r*dt);
        // initialize prices array
        std::vector<U> prices(N+1);
        #pragma omd vectorize for
        for (int i = 0; i <= N; ++i) {
            prices[i] = std::max(o.K - static_cast<U>(dynamics.S * std::pow(u, i) * std::pow(d, N-i)), static_cast<U>(0.0));
        }
        // calculate each step:
        #pragma omd vectorize for
        for (int t = N-1; t >= 0; --t) {
            #pragma omd vectorize for
            for (int i = 0; i <= t; ++i) {
                prices[i] = half_discount * (prices[i+1] + prices[i]);
            }
        }
        return prices[0];
    }
    int N; // N >= 1000
};

} // end namespace part1

inline void
run_part1()
{
    using namespace part1;
    std::cout << "[Part 1] Starting..." << std::endl;
    CallOption<float> call = CallOption<float>{1, 100};;
    PutOption<float> put = PutOption<float>{1, 100};
    std::vector<Dynamics<float>> dynamics_vector;
    dynamics_vector.resize(5);
    for (size_t i = 0; i < 5; ++i) {
        dynamics_vector[i] = Dynamics<float>{static_cast<float>(90 + i*5), 0.03f, 0.3f}; // S = 90, 95, 100, 105, 110
    }
    
    int N = 1000;
    AnalyticsEngine<float> engine = AnalyticsEngine<float>(N);

    float sum_call = 0.0f;
    float sum_put = 0.0f;
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omd parallel for
    for (size_t i = 0; i < dynamics_vector.size(); ++i) {
        float call_price = engine.priceEuroCall_JarrowRudd(call, dynamics_vector[i]);
        std::cout << call_price << std::endl;
        sum_call += call_price;
        sum_put += engine.priceEuroPut_JarrowRudd(put, dynamics_vector[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << "\n";
    std::cout << "sum call= " << sum_call << "\n" << "sum put= " << sum_put << "\n";
    std::cout << "[Part 1] Ended.\n" << std::endl;
}

// PART 2

using DataRow = std::array<float, 5>;
using DataMatrix = std::vector<DataRow>;

float random_data(float low, float hi)
{
    float r = (float)rand() / (float)RAND_MAX;
    return low + r * (hi - low);
}

std::unique_ptr<DataMatrix>
random_data_matrix()
{
    auto data_matrix = std::make_unique<DataMatrix>(10'000);
    DataRow dataRow; // K, T, S, r, v
    for (size_t row = 0; row < data_matrix->size(); ++row) {
        dataRow[0] = random_data(90, 110);
        dataRow[1] = random_data(0.75, 1.25);
        dataRow[2] = random_data(80, 120);
        dataRow[3] = random_data(0.01, 0.05);
        dataRow[4] = random_data(0.2, 0.4);
        // dataRow[0] = 100.0f;
        // dataRow[1] = 1.0f;
        // dataRow[2] = 90.0f;
        // dataRow[3] = 0.03f;
        // dataRow[4] = 0.3f;
        (*data_matrix)[row] = dataRow;
    }
    return data_matrix;
}

size_t constexpr N = 1'000;

inline float
pow(float const& value, int const& p) noexcept
{
    float result = value;
    #pragma omp simd
    for (int i = 0; i < p-1; ++i) {
        result*=value;
    }
    return result;
}

inline float
max_or_zero(float x) noexcept
{
    union {
        float f;
        uint32_t u;
    } tmp;
    tmp.f = x;
    int is_negative = tmp.u >> 31;
    return x * (1 - is_negative);
}

struct alignas(64) ThreadLocalData {
    std::array<float, N+1> prices;
    float dt;
    float half_discount;
    float base;
    float diff;
    float u;
    float d;
    int block_size;
    DataMatrix const& input_matrix;
    float& partial_sum;
    ThreadLocalData(int bs, DataMatrix const& in_mat, float &ps): block_size(bs), input_matrix(in_mat), partial_sum(ps) {}
};

void
thread_fn(ThreadLocalData& thread_data)
{
    #pragma omp simd
    for (int m = 0; m < thread_data.block_size; ++m) {
        // K 0 (*input_matrix)[i][0]; // K = (*input_matrix)[i][0]; // K=100.0f;
        // T 1 (*input_matrix)[i][1]; // T = (*input_matrix)[i][1]; // T=1.0f;
        // S 2 (*input_matrix)[i][2]; // S = (*input_matrix)[i][2]; // S=90.0f+static_cast<float>(i*5);
        // r 3 (*input_matrix)[i][3]; // r = (*input_matrix)[i][3]; // r=0.03f;
        // v 4 (*input_matrix)[i][4]; // v = (*input_matrix)[i][4]; // v=0.3f;
        float const r = thread_data.input_matrix[m][3];
        float const v = thread_data.input_matrix[m][4];
        thread_data.dt = thread_data.input_matrix[m][1]/N;
        thread_data.half_discount = 0.5*std::exp(-r*thread_data.dt);
        thread_data.base = (r - 0.5f * v * v) * thread_data.dt;
        thread_data.diff = v * std::sqrt(thread_data.dt);
        thread_data.u = std::exp(thread_data.base + thread_data.diff);
        thread_data.d = std::exp(thread_data.base - thread_data.diff);
        #pragma omd simd
        for (size_t i = 0; i <= N; ++i) {
            thread_data.prices[i] = max_or_zero(thread_data.input_matrix[m][2] * static_cast<float>(pow(thread_data.u, i)) * static_cast<float>(pow(thread_data.d, N-i)) - thread_data.input_matrix[m][0]);
        }
        #pragma omd vectorize for
        for (int t = N-1; t >= 0; --t) {
            #pragma omd simd
            for (size_t i = 0; i <= t; ++i) {
                thread_data.prices[i] = thread_data.half_discount*(thread_data.prices[i+1] + thread_data.prices[i]); // Wait to discount until after.
            }
        }
        thread_data.partial_sum += thread_data.prices[0];
    }
}

inline void
time_million_calls()
{
    std::cout << "[Part 2] Starting..." << std::endl;
    std::unique_ptr<DataMatrix> input_matrix = random_data_matrix();
    int const N_THREADS = std::thread::hardware_concurrency();
    float sum = 0.0f;
    int EPOCHS = static_cast<int>(input_matrix->size());
    int block_size = EPOCHS / N_THREADS;
    int remainder = EPOCHS % N_THREADS;
    std::vector<int> block_sizes(N_THREADS);
    for (int i = 0; i < N_THREADS; ++i) {
        block_sizes[i] = block_size + (i < remainder ? 1 : 0);
    }
    std::vector<std::unique_ptr<ThreadLocalData>> thread_data_storage(N_THREADS);
    std::vector<float> partial_sums(N_THREADS, 0.0f);

    std::vector<DataMatrix> spliced_inputs(N_THREADS);
    for (int i = 0; i < N_THREADS; ++i) {
        spliced_inputs[i].resize(block_sizes[i]);
        for (int j = 0; j < block_sizes[i]; ++j) {
            spliced_inputs[i][j] = (*input_matrix)[i * block_size + j];
        }
        thread_data_storage[i] = std::make_unique<ThreadLocalData>(block_size, std::cref(spliced_inputs[i]), std::ref(partial_sums[i]));
    }
    std::vector<std::thread> threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_THREADS; ++i) {
        threads.emplace_back(thread_fn, std::ref(*thread_data_storage[i]));
    }
    for (auto& thread: threads) {
        thread.join();
    }
    // for (auto &data : thread_data_storage) {
    //     sum += data->partial_sum;
    // }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;
    // std::cout << "sum= " << sum << std::endl;
    std::cout << "[Part 2] Ended.\n" << std::endl;
}

int
main()
{
    // run_part1();

    std::cout << std::endl;
    // 1'000'000 EU Call Option Pricing Runtime
    time_million_calls();
}
