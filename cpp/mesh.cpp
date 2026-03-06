#include "mesh.h"
#include "tree.h"
#include "vertex.h"
#include "edge.h"
#include "face.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <numeric>
#include <limits>

DecisionMesh::DecisionMesh(const std::vector<double>& x_data,
                           const std::vector<double>& y_data,
                           const std::vector<double>& z_data)
    : rng(42)
{
    n_points = (int)x_data.size();
    X.resize(n_points * 2);
    values = z_data;
    for (int i = 0; i < n_points; ++i) {
        X[i * 2 + 0] = x_data[i];
        X[i * 2 + 1] = y_data[i];
    }

    root = make_node();
    leaves.insert(root);

    create_outer_vertices();
    create_outer_edges();

    split_edge = make_edge(outer_vertices["bottom_left"],
                           outer_vertices["top_right"], true);

    // Compute mask: X @ split_edge.normal >= split_edge.intercept
    std::vector<bool> mask_pos(n_points), mask_neg(n_points);
    for (int i = 0; i < n_points; ++i) {
        double val = split_edge->normal[0] * get_x(i, 0) +
                     split_edge->normal[1] * get_x(i, 1);
        mask_pos[i] = (val >= split_edge->intercept);
        mask_neg[i] = !mask_pos[i];
    }

    top_face = make_face(split_edge, outer_edges["left"], outer_edges["top"],
                         mask_pos, true, "+");
    bottom_face = make_face(split_edge, outer_edges["bottom"], outer_edges["right"],
                            mask_neg, true, "-");
    root->split(split_edge, top_face, bottom_face);

    for (Vertex* v : std::set<Vertex*>(vertices)) {
        v->update_info();
    }
}

DecisionMesh::~DecisionMesh() {
    for (auto* p : all_nodes) delete p;
    for (auto* p : all_vertices) delete p;
    for (auto* p : all_edges) delete p;
    for (auto* p : all_faces) delete p;
}

TreeNode* DecisionMesh::make_node(TreeNode* parent, Face* face) {
    auto* n = new TreeNode(this, parent, face);
    all_nodes.push_back(n);
    return n;
}

Vertex* DecisionMesh::make_vertex(double x, double y, bool active,
                                   Edge* parent_edge, bool real_vertex) {
    auto* v = new Vertex(this, x, y, active, parent_edge, real_vertex);
    all_vertices.push_back(v);
    return v;
}

Edge* DecisionMesh::make_edge(Vertex* v0, Vertex* v1, bool active) {
    auto* e = new Edge(this, v0, v1, active);
    all_edges.push_back(e);
    return e;
}

Face* DecisionMesh::make_face(Edge* e0, Edge* e1, Edge* e2,
                               const std::vector<bool>& mask, bool active,
                               const std::string& path) {
    auto* f = new Face(this, e0, e1, e2, mask, active, path);
    all_faces.push_back(f);
    return f;
}

// --- Heap operations using std::map<double, std::set<Vertex*>> ---

void DecisionMesh::heap_set(Vertex* v, double neg_loss) {
    // Remove old entry if exists
    heap_pop(v);
    loss_heap[neg_loss].insert(v);
}

void DecisionMesh::heap_pop(Vertex* v) {
    // Linear scan to find and remove - could be optimized
    for (auto it = loss_heap.begin(); it != loss_heap.end(); ) {
        it->second.erase(v);
        if (it->second.empty()) {
            it = loss_heap.erase(it);
        } else {
            ++it;
        }
    }
}

std::pair<Vertex*, double> DecisionMesh::heap_peek() const {
    if (loss_heap.empty()) return {nullptr, 0.0};
    auto it = loss_heap.begin();  // smallest key = most negative = best
    Vertex* v = *it->second.begin();
    return {v, it->first};
}

Face* DecisionMesh::random_face() {
    std::vector<Face*> faces(active_faces.begin(), active_faces.end());
    if (faces.empty()) return nullptr;

    std::vector<double> weights(faces.size());
    for (size_t i = 0; i < faces.size(); ++i) {
        double a = faces[i]->area();
        int count = 0;
        for (int j = 0; j < n_points; ++j) {
            if (faces[i]->mask[j]) ++count;
        }
        weights[i] = (a > 0 && std::isfinite(a)) ? a * count : 0.0;
    }

    double total = 0;
    for (double w : weights) total += w;

    if (total <= 0) {
        std::uniform_int_distribution<int> dist(0, (int)faces.size() - 1);
        return faces[dist(rng)];
    }

    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng);
    double cum = 0;
    for (size_t i = 0; i < faces.size(); ++i) {
        cum += weights[i];
        if (r <= cum) return faces[i];
    }
    return faces.back();
}

void DecisionMesh::update_best_vertex(double random_prob) {
    Vertex* best = nullptr;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (random_prob > 0.0 && dist(rng) < random_prob) {
        Face* face = random_face();
        if (!face) return;

        // Pick longest edge
        double max_len = 0;
        Edge* longest = nullptr;
        for (int i = 0; i < 3; ++i) {
            if (face->edges[i]->length > max_len) {
                max_len = face->edges[i]->length;
                longest = face->edges[i];
            }
        }
        if (longest) best = longest->midpoint;
    } else {
        auto [v, _] = heap_peek();
        best = v;
    }

    if (!best) return;

    if (best->active) {
        best->update_height();
    } else {
        best->activate();
    }
}

void DecisionMesh::create_outer_vertices() {
    xmin = X[0]; xmax = X[0];
    ymin = X[1]; ymax = X[1];
    for (int i = 0; i < n_points; ++i) {
        double xi = get_x(i, 0), yi = get_x(i, 1);
        if (xi < xmin) xmin = xi;
        if (xi > xmax) xmax = xi;
        if (yi < ymin) ymin = yi;
        if (yi > ymax) ymax = yi;
    }
    outer_vertices["bottom_left"]  = make_vertex(xmin, ymin, true);
    outer_vertices["top_left"]     = make_vertex(xmin, ymax, true);
    outer_vertices["bottom_right"] = make_vertex(xmax, ymin, true);
    outer_vertices["top_right"]    = make_vertex(xmax, ymax, true);
}

void DecisionMesh::create_outer_edges() {
    outer_edges["left"]   = make_edge(outer_vertices["top_left"],    outer_vertices["bottom_left"], false);
    outer_edges["bottom"] = make_edge(outer_vertices["bottom_right"],outer_vertices["bottom_left"], false);
    outer_edges["right"]  = make_edge(outer_vertices["top_right"],   outer_vertices["bottom_right"], false);
    outer_edges["top"]    = make_edge(outer_vertices["top_right"],   outer_vertices["top_left"], false);
}

void DecisionMesh::write_svg(const std::string& filename, double w, double h) const {
    // Find height range for coloring
    double hmin = 1e300, hmax = -1e300;
    for (Vertex* v : vertices) {
        if (v->height < hmin) hmin = v->height;
        if (v->height > hmax) hmax = v->height;
    }
    if (hmax - hmin < 1e-12) { hmin -= 1; hmax += 1; }

    double padx = 20, pady = 20;
    double sx = (w - 2 * padx) / (xmax - xmin);
    double sy = (h - 2 * pady) / (ymax - ymin);
    double scale = std::min(sx, sy);

    auto tx = [&](double x) { return padx + (x - xmin) * scale; };
    auto ty = [&](double y) { return h - pady - (y - ymin) * scale; };

    // Simple blue-white-red colormap
    auto color = [&](double val) -> std::string {
        double t = (val - hmin) / (hmax - hmin);
        t = std::max(0.0, std::min(1.0, t));
        int r, g, b;
        if (t < 0.5) {
            double s = t * 2;
            r = (int)(68 + s * (255 - 68));
            g = (int)(1 + s * (255 - 1));
            b = (int)(84 + s * (255 - 84));
        } else {
            double s = (t - 0.5) * 2;
            r = (int)(255 - s * (255 - 253));
            g = (int)(255 - s * (255 - 231));
            b = (int)(255 - s * (255 - 37));
        }
        char buf[16];
        snprintf(buf, sizeof(buf), "#%02x%02x%02x", r, g, b);
        return buf;
    };

    std::ofstream out(filename);
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w
        << "\" height=\"" << h << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

    // Draw filled triangles
    for (Face* f : active_faces) {
        double avg_h = (f->vertices[0]->height + f->vertices[1]->height + f->vertices[2]->height) / 3.0;
        out << "<polygon points=\""
            << tx(f->vertices[0]->x) << "," << ty(f->vertices[0]->y) << " "
            << tx(f->vertices[1]->x) << "," << ty(f->vertices[1]->y) << " "
            << tx(f->vertices[2]->x) << "," << ty(f->vertices[2]->y) << "\" "
            << "fill=\"" << color(avg_h) << "\" stroke=\"black\" stroke-width=\"0.5\"/>\n";
    }

    out << "</svg>\n";
    out.close();
}
