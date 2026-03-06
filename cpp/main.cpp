#include "mesh.h"
#include "face.h"
#include "vertex.h"
#include "edge.h"
#include "tree.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

static double compute_mse(DecisionMesh& mesh) {
    double total_sse = 0.0;
    int total_n = 0;
    for (Face* face : mesh.active_faces) {
        if (face->n_covered == 0) continue;
        Vertex* v0 = face->vertices[0];
        Vertex* v1 = face->vertices[1];
        Vertex* v2 = face->vertices[2];

        // Solve 3x3 system for affine model z = a*x + b*y + c
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

int main(int argc, char** argv) {
    int n_points = 200000;
    int time_limit_sec = 90;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            n_points = std::atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            time_limit_sec = std::atoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: decision_mesh [-n points] [-t seconds] [-i input.csv]\n");
            printf("  -n  Number of generated data points (default: 200000)\n");
            printf("  -t  Time limit for refinement in seconds (default: 90)\n");
            printf("  -i  Input CSV file (x,y,z columns). If omitted, generates demo data.\n");
            return 0;
        }
    }

    std::vector<double> x_data, y_data, z_data;

    if (!csv_path.empty()) {
        printf("Loading CSV from %s...\n", csv_path.c_str());
        if (!load_csv(csv_path, x_data, y_data, z_data)) {
            fprintf(stderr, "Failed to load CSV file.\n");
            return 1;
        }
        n_points = (int)x_data.size();
        printf("Loaded %d points.\n", n_points);
    } else {
        printf("Generating %d demo data points...\n", n_points);
        generate_demo_data(n_points, x_data, y_data, z_data);
    }

    printf("\nRunning for %d seconds with and without Empirical Bayes...\n\n", time_limit_sec);
    printf("%-25s %10s %10s %8s %8s %12s\n", "Method", "Build (s)", "Refine(s)", "Steps", "Faces", "MSE");
    printf("-------------------------------------------------------------------------------\n");

    auto run_benchmark = [&](const char* label, bool use_eb) {
        Vertex::_seq = 0; Edge::_seq = 0; Face::_seq = 0; TreeNode::_seq = 0;

        auto t0 = std::clock();
        DecisionMesh mesh(x_data, y_data, z_data, use_eb);
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

        double mse = compute_mse(mesh);
        printf("%-25s %10.1f %10.1f %8d %8d %12.6f\n",
               label, build_time, refine_time,
               iters, (int)mesh.active_faces.size(), mse);
    };

    run_benchmark("No regularization", false);
    run_benchmark("Empirical Bayes", true);

    printf("\nDone.\n");
    return 0;
}
