#pragma once
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

struct TreeNode;
struct Vertex;
struct Edge;
struct Face;

struct TimingRecord {
    int iteration;
    double find_best_us;   // microseconds spent finding best vertex
    double split_us;       // microseconds spent splitting/activating
    double update_info_us; // microseconds spent in update_info propagation
    double total_us;       // total for this refinement
    int active_faces;
    int n_vertices;
    // Diagnostic counters for understanding cascade behavior
    int cascade_activations;  // how many recursive vertex activations (completions)
    int faces_created;        // faces created during this refinement
    int edges_created;        // edges created during this refinement
    int add_face_calls;       // Edge::add_face calls (each does coord computation)
    int update_info_calls;    // Vertex::update_info calls
};

// Thread-local counters for cascade instrumentation
struct CascadeCounters {
    int cascade_activations = 0;
    int faces_created = 0;
    int edges_created = 0;
    int add_face_calls = 0;
    int update_info_calls = 0;
};

// Global counters (single-threaded, so this is fine)
extern CascadeCounters g_counters;

struct DecisionMesh {
    double max_aspect_ratio = 5.0;

    // Data: n rows, 2 columns for X, 1 column for values
    int n_points = 0;
    std::vector<double> X;       // n_points * 2, row-major
    std::vector<double> values;  // n_points

    double xmin, xmax, ymin, ymax;

    // Object ownership (heap-allocated, cleaned up in destructor)
    std::vector<TreeNode*> all_nodes;
    std::vector<Vertex*> all_vertices;
    std::vector<Edge*> all_edges;
    std::vector<Face*> all_faces;

    TreeNode* root = nullptr;
    std::set<TreeNode*> leaves;
    std::set<Face*> active_faces;
    std::set<Vertex*> vertices;      // "real" vertices
    std::set<Edge*> active_edges;

    // Loss heap: map from negative loss_reduction -> vertex
    // We use a std::map<double, std::set<Vertex*>> for decrease-key support
    // Or simpler: just track in vertices and scan. For performance, use a map.
    std::map<double, std::set<Vertex*>> loss_heap;

    // Outer geometry
    std::map<std::string, Vertex*> outer_vertices;
    std::map<std::string, Edge*> outer_edges;
    Edge* split_edge = nullptr;
    Face* top_face = nullptr;
    Face* bottom_face = nullptr;

    std::mt19937 rng;

    DecisionMesh(const std::vector<double>& x_data,
                 const std::vector<double>& y_data,
                 const std::vector<double>& z_data);
    ~DecisionMesh();

    // Factory methods (own the memory)
    TreeNode* make_node(TreeNode* parent = nullptr, Face* face = nullptr);
    Vertex* make_vertex(double x, double y, bool active = false,
                        Edge* parent_edge = nullptr, bool real_vertex = true);
    Edge* make_edge(Vertex* v0, Vertex* v1, bool active);
    Face* make_face(Edge* e0, Edge* e1, Edge* e2,
                    const std::vector<bool>& mask, bool active = true,
                    const std::string& path = "", bool skip_coords = false);

    // Heap operations
    void heap_set(Vertex* v, double neg_loss);
    void heap_pop(Vertex* v);
    std::pair<Vertex*, double> heap_peek() const;

    // Core
    void create_outer_vertices();
    void create_outer_edges();

    Face* random_face();
    void update_best_vertex(double random_prob = 0.0);
    TimingRecord update_best_vertex_timed(double random_prob, int iteration);

    double get_x(int i, int col) const { return X[i * 2 + col]; }

    // Export mesh to SVG (simple visualization)
    void write_svg(const std::string& filename, double width = 800, double height = 800) const;
};
