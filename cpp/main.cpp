#include "mesh.h"
#include "face.h"
#include "vertex.h"
#include "edge.h"
#include "tree.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

static void generate_demo_data(int n, std::vector<double>& x,
                               std::vector<double>& y, std::vector<double>& z) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> unif(-4.0, 4.0);
    std::normal_distribution<double> norm(0.0, 1.0);

    x.resize(n);
    y.resize(n);
    z.resize(n);
    for (int i = 0; i < n; ++i) {
        x[i] = unif(gen);
        y[i] = unif(gen);
        z[i] = norm(gen) + 2.0 * std::cos(5.0 * x[i]) * std::cos(2.0 * y[i]);
    }
}

static bool load_csv(const std::string& path, std::vector<double>& x,
                     std::vector<double>& y, std::vector<double>& z) {
    std::ifstream in(path);
    if (!in.is_open()) return false;

    std::string line;
    if (std::getline(in, line)) {
        std::istringstream test(line);
        double tmp;
        char comma;
        if (!(test >> tmp >> comma >> tmp >> comma >> tmp)) {
            // Header line, skip it
        } else {
            std::istringstream ss(line);
            double a, b, c;
            char c1, c2;
            ss >> a >> c1 >> b >> c2 >> c;
            x.push_back(a);
            y.push_back(b);
            z.push_back(c);
        }
    }
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double a, b, c;
        char c1, c2;
        if (ss >> a >> c1 >> b >> c2 >> c) {
            x.push_back(a);
            y.push_back(b);
            z.push_back(c);
        }
    }
    return !x.empty();
}

// Compute in-sample MSE by iterating over face-assigned points
static double compute_train_mse(DecisionMesh& mesh) {
    double total_sse = 0.0;
    int total_n = 0;
    for (Face* face : mesh.active_faces) {
        if (face->n_covered == 0) continue;
        Vertex* v0 = face->vertices[0];
        Vertex* v1 = face->vertices[1];
        Vertex* v2 = face->vertices[2];

        double A[3][3] = {
            {v0->x, v0->y, 1.0},
            {v1->x, v1->y, 1.0},
            {v2->x, v2->y, 1.0}
        };
        double h[3] = {v0->height, v1->height, v2->height};

        double det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                   - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                   + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

        if (std::abs(det) < 1e-12) {
            double avg_h = (h[0] + h[1] + h[2]) / 3.0;
            for (int idx : face->coords_indices) {
                double diff = mesh.values[idx] - avg_h;
                total_sse += diff * diff;
            }
        } else {
            double inv_det = 1.0 / det;
            double a_ = inv_det * (h[0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                                 - A[0][1]*(h[1]*A[2][2] - A[1][2]*h[2])
                                 + A[0][2]*(h[1]*A[2][1] - A[1][1]*h[2]));
            double b_ = inv_det * (A[0][0]*(h[1]*A[2][2] - A[1][2]*h[2])
                                 - h[0]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                                 + A[0][2]*(A[1][0]*h[2] - h[1]*A[2][0]));
            double c_ = inv_det * (A[0][0]*(A[1][1]*h[2] - h[1]*A[2][1])
                                 - A[0][1]*(A[1][0]*h[2] - h[1]*A[2][0])
                                 + h[0]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]));

            for (int idx : face->coords_indices) {
                double px = mesh.get_x(idx, 0);
                double py = mesh.get_x(idx, 1);
                double pred = a_ * px + b_ * py + c_;
                double diff = mesh.values[idx] - pred;
                total_sse += diff * diff;
            }
        }
        total_n += face->n_covered;
    }
    return (total_n > 0) ? total_sse / total_n : 0.0;
}

// Compute out-of-sample MSE using tree-based predict (O(log n) per point)
static double compute_test_mse(DecisionMesh& mesh,
                               const std::vector<double>& tx,
                               const std::vector<double>& ty,
                               const std::vector<double>& tz) {
    int n = (int)tx.size();
    double sse = 0.0;
    for (int i = 0; i < n; ++i) {
        double pred = mesh.predict(tx[i], ty[i]);
        double diff = tz[i] - pred;
        sse += diff * diff;
    }
    return sse / n;
}

int main(int argc, char** argv) {
    int n_points = 200000;
    int time_limit_sec = 90;
    double test_frac = 0.2;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            n_points = std::atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            time_limit_sec = std::atoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "-f" && i + 1 < argc) {
            test_frac = std::atof(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: decision_mesh [-n points] [-t seconds] [-f test_frac] [-i input.csv]\n");
            printf("  -n  Number of generated data points (default: 200000)\n");
            printf("  -t  Time limit for refinement in seconds (default: 90)\n");
            printf("  -f  Fraction of data held out for testing (default: 0.2)\n");
            printf("  -i  Input CSV file (x,y,z columns). If omitted, generates demo data.\n");
            return 0;
        }
    }

    std::vector<double> x_all, y_all, z_all;

    if (!csv_path.empty()) {
        printf("Loading CSV from %s...\n", csv_path.c_str());
        if (!load_csv(csv_path, x_all, y_all, z_all)) {
            fprintf(stderr, "Failed to load CSV file.\n");
            return 1;
        }
        n_points = (int)x_all.size();
        printf("Loaded %d points.\n", n_points);
    } else {
        printf("Generating %d demo data points...\n", n_points);
        generate_demo_data(n_points, x_all, y_all, z_all);
    }

    // --- Train/test split ---
    int n_test = (int)(n_points * test_frac);
    int n_train = n_points - n_test;

    // Shuffle indices for random split
    std::vector<int> indices(n_points);
    for (int i = 0; i < n_points; ++i) indices[i] = i;
    std::mt19937 shuffle_rng(123);
    std::shuffle(indices.begin(), indices.end(), shuffle_rng);

    std::vector<double> x_train(n_train), y_train(n_train), z_train(n_train);
    std::vector<double> x_test(n_test), y_test(n_test), z_test(n_test);
    for (int i = 0; i < n_train; ++i) {
        int idx = indices[i];
        x_train[i] = x_all[idx];
        y_train[i] = y_all[idx];
        z_train[i] = z_all[idx];
    }
    for (int i = 0; i < n_test; ++i) {
        int idx = indices[n_train + i];
        x_test[i] = x_all[idx];
        y_test[i] = y_all[idx];
        z_test[i] = z_all[idx];
    }

    printf("Train: %d points, Test: %d points (%.0f%% held out)\n",
           n_train, n_test, test_frac * 100);
    printf("\nRunning for %d seconds with and without Empirical Bayes...\n\n", time_limit_sec);
    printf("%-25s %10s %10s %8s %8s %12s %12s %12s\n",
           "Method", "Build (s)", "Refine(s)", "Steps", "Faces",
           "Train MSE", "Test MSE", "Pred (us/pt)");
    printf("----------------------------------------------------------------------------------------------------\n");

    auto run_benchmark = [&](const char* label, bool use_eb, const std::string& prefix) {
        Vertex::_seq = 0; Edge::_seq = 0; Face::_seq = 0; TreeNode::_seq = 0;

        auto t0 = std::clock();
        DecisionMesh mesh(x_train, y_train, z_train, use_eb);
        double build_time = (double)(std::clock() - t0) / CLOCKS_PER_SEC;

        auto start = std::clock();
        int iters = 0;
        while (true) {
            double elapsed = (double)(std::clock() - start) / CLOCKS_PER_SEC;
            if (elapsed >= time_limit_sec) break;
            mesh.update_best_vertex(0.0);
            iters++;
            if (iters % 2000 == 0) {
                elapsed = (double)(std::clock() - start) / CLOCKS_PER_SEC;
                fprintf(stderr, "  [%s] %d steps, %.0fs elapsed...\n", label, iters, elapsed);
            }
        }
        double refine_time = (double)(std::clock() - start) / CLOCKS_PER_SEC;

        double train_mse = compute_train_mse(mesh);

        // Out-of-sample prediction via tree walk: O(log faces) per point
        using hrclock = std::chrono::high_resolution_clock;
        auto pred_t0 = hrclock::now();
        double test_mse = compute_test_mse(mesh, x_test, y_test, z_test);
        auto pred_t1 = hrclock::now();
        double pred_us_per_pt = std::chrono::duration<double, std::micro>(pred_t1 - pred_t0).count()
                                / n_test;

        printf("%-25s %10.1f %10.1f %8d %8d %12.6f %12.6f %12.3f\n",
               label, build_time, refine_time,
               iters, (int)mesh.active_faces.size(),
               train_mse, test_mse, pred_us_per_pt);

        // Write mesh data for visualization
        mesh.write_mesh_csv(prefix);
        fprintf(stderr, "  Wrote %s_vertices.csv, %s_triangles.csv\n",
                prefix.c_str(), prefix.c_str());
    };

    run_benchmark("No regularization", false, "mesh_noreg");
    run_benchmark("Empirical Bayes", true, "mesh_eb");

    printf("\nDone.\n");
    return 0;
}
