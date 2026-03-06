#include "mesh.h"
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
    // Skip header if present
    if (std::getline(in, line)) {
        // Check if first line is numeric
        std::istringstream test(line);
        double tmp;
        char comma;
        if (!(test >> tmp >> comma >> tmp >> comma >> tmp)) {
            // Header line, skip it
        } else {
            // First line is data, parse it
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

int main(int argc, char** argv) {
    int n_points = 200000;
    int time_limit_sec = 90;
    std::string csv_path;
    std::string output_svg = "mesh_output.svg";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            n_points = std::atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            time_limit_sec = std::atoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_svg = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: decision_mesh [-n points] [-t seconds] [-i input.csv] [-o output.svg]\n");
            printf("  -n  Number of generated data points (default: 200000)\n");
            printf("  -t  Time limit for refinement in seconds (default: 90)\n");
            printf("  -i  Input CSV file (x,y,z columns). If omitted, generates demo data.\n");
            printf("  -o  Output SVG file (default: mesh_output.svg)\n");
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

    printf("Building mesh...\n");
    auto t0 = std::clock();
    DecisionMesh mesh(x_data, y_data, z_data);
    double build_time = (double)(std::clock() - t0) / CLOCKS_PER_SEC;
    printf("Mesh built in %.1fs.\n", build_time);

    printf("Refining for up to %d seconds...\n", time_limit_sec);
    auto start = std::clock();
    int iters = 0;
    while (true) {
        double elapsed = (double)(std::clock() - start) / CLOCKS_PER_SEC;
        if (elapsed >= time_limit_sec) break;

        mesh.update_best_vertex(0.1);
        iters++;

        if (iters % 50 == 0) {
            elapsed = (double)(std::clock() - start) / CLOCKS_PER_SEC;
            printf("\r%d refinements, %.0fs elapsed", iters, elapsed);
            fflush(stdout);
        }
    }
    double total = (double)(std::clock() - start) / CLOCKS_PER_SEC;
    printf("\nDone: %d refinements in %.1fs (%.0fms/ref)\n",
           iters, total, total / iters * 1000);

    printf("Writing SVG to %s...\n", output_svg.c_str());
    mesh.write_svg(output_svg);

    printf("Active faces: %d\n", (int)mesh.active_faces.size());
    printf("Vertices: %d\n", (int)mesh.vertices.size());
    printf("Done.\n");

    return 0;
}
