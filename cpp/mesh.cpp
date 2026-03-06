#include "mesh.h"
#include "tree.h"
#include "vertex.h"
#include "edge.h"
#include "face.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <numeric>
#include <limits>

CascadeCounters g_counters;

DecisionMesh::DecisionMesh(const std::vector<double>& x_data,
                           const std::vector<double>& y_data,
                           const std::vector<double>& z_data,
                           bool use_eb_flag)
    : rng(42), use_eb(use_eb_flag)
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

    // Bootstrap empirical Bayes: compute tau_sq from initial estimates,
    // then re-update non-corner vertices with regularization
    if (use_eb) {
        recompute_tau_sq();
        for (Vertex* v : std::set<Vertex*>(vertices)) {
            if (v->parent_edge != nullptr) {
                v->update_info();
            }
        }
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
    g_counters.edges_created++;
    auto* e = new Edge(this, v0, v1, active);
    all_edges.push_back(e);
    return e;
}

Face* DecisionMesh::make_face(Edge* e0, Edge* e1, Edge* e2,
                               const std::vector<bool>& mask, bool active,
                               const std::string& path, bool skip_coords) {
    g_counters.faces_created++;
    auto* f = new Face(this, e0, e1, e2, mask, active, path, skip_coords);
    all_faces.push_back(f);
    return f;
}

// --- Heap operations using std::map<double, std::set<Vertex*>> ---

void DecisionMesh::heap_set(Vertex* v, double neg_loss) {
    // O(log n) removal of old entry using cached key
    if (v->in_heap) {
        auto it = loss_heap.find(v->heap_key);
        if (it != loss_heap.end()) {
            it->second.erase(v);
            if (it->second.empty()) {
                loss_heap.erase(it);
            }
        }
        v->in_heap = false;
    }
    // O(log n) insertion at new key
    loss_heap[neg_loss].insert(v);
    v->heap_key = neg_loss;
    v->in_heap = true;
}

void DecisionMesh::heap_pop(Vertex* v) {
    if (!v->in_heap) return;
    auto it = loss_heap.find(v->heap_key);
    if (it != loss_heap.end()) {
        it->second.erase(v);
        if (it->second.empty()) {
            loss_heap.erase(it);
        }
    }
    v->in_heap = false;
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
        int count = faces[i]->n_covered;
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

void DecisionMesh::update_best_vertex(double) {
    if (use_eb) {
        maybe_recompute_tau_sq();
    }

    auto [best, _] = heap_peek();
    if (!best) return;

    if (best->active) {
        best->update_height();
    } else {
        best->activate();
    }
}

TimingRecord DecisionMesh::update_best_vertex_timed(double random_prob, int iteration) {
    using clock = std::chrono::high_resolution_clock;
    TimingRecord rec{};
    rec.iteration = iteration;

    if (use_eb) {
        maybe_recompute_tau_sq();
    }

    // Phase 1: Find best vertex (heap peek is O(1))
    auto t0 = clock::now();
    auto [best, _] = heap_peek();
    auto t1 = clock::now();
    rec.find_best_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    if (!best) {
        rec.total_us = rec.find_best_us;
        rec.active_faces = (int)active_faces.size();
        rec.n_vertices = (int)vertices.size();
        return rec;
    }

    // Phase 2: Split/activate — reset cascade counters
    g_counters = CascadeCounters{};
    auto t2 = clock::now();
    if (best->active) {
        best->height = best->new_height;
        if (best->parent_edge != nullptr) {
            best->delta_pooled = best->height - best->mu_lin();
        }
        auto t2b = clock::now();
        std::set<Vertex*> to_update(best->affected_vertices);
        if (use_eb) {
            to_update.insert(best->prior_children.begin(), best->prior_children.end());
        }
        for (Vertex* v : to_update) {
            v->update_info();
        }
        best->loss_reduction = 0;
        heap_set(best, 0);
        auto t3 = clock::now();
        rec.split_us = std::chrono::duration<double, std::micro>(t2b - t2).count();
        rec.update_info_us = std::chrono::duration<double, std::micro>(t3 - t2b).count();
    } else {
        // activate: set height, split parent edge (creates new faces), then update_info
        best->active = true;
        best->height = best->new_height;

        // Set EB posterior fields before split
        if (use_eb && best->parent_edge != nullptr) {
            best->delta_pooled = best->height - best->mu_lin();
            double inv_sigma = (best->sigma_sq < 1e300) ? (1.0 / best->sigma_sq) : 0.0;
            double xTx_reg = inv_sigma + best->lambda_v;
            best->sigma_pooled = (xTx_reg > 0) ? (1.0 / xTx_reg) : 1e300;
        }

        best->parent_edge->split();
        auto t2b = clock::now();
        std::set<Vertex*> to_update(best->affected_vertices);
        if (use_eb) {
            to_update.insert(best->prior_children.begin(), best->prior_children.end());
        }
        for (Vertex* v : to_update) {
            v->update_info();
        }
        best->loss_reduction = 0;
        heap_set(best, 0);
        auto t3 = clock::now();
        rec.split_us = std::chrono::duration<double, std::micro>(t2b - t2).count();
        rec.update_info_us = std::chrono::duration<double, std::micro>(t3 - t2b).count();
    }

    auto tend = clock::now();
    rec.total_us = std::chrono::duration<double, std::micro>(tend - t0).count();
    rec.active_faces = (int)active_faces.size();
    rec.n_vertices = (int)vertices.size();
    rec.cascade_activations = g_counters.cascade_activations;
    rec.faces_created = g_counters.faces_created;
    rec.edges_created = g_counters.edges_created;
    rec.add_face_calls = g_counters.add_face_calls;
    rec.update_info_calls = g_counters.update_info_calls;
    return rec;
}

// --- Empirical Bayes ---

void DecisionMesh::recompute_tau_sq() {
    // Group non-corner vertices by depth
    std::map<int, std::vector<Vertex*>> by_depth;
    for (Vertex* v : vertices) {
        if (v->parent_edge == nullptr) continue;
        if (v->sigma_sq >= 1e300 || v->sigma_sq <= 0) continue;
        by_depth[v->depth].push_back(v);
    }

    for (auto& [d, verts] : by_depth) {
        int m = (int)verts.size();
        tau_sq_vertex_counts[d] = m;

        if (m < 3) continue;

        double sum_w = 0.0;
        double sum_wd = 0.0;
        for (Vertex* v : verts) {
            double w = 1.0 / v->sigma_sq;
            sum_w += w;
            sum_wd += v->delta_pooled * w;
        }
        if (sum_w <= 0) continue;

        double mu_d = sum_wd / sum_w;
        mu_delta[d] = mu_d;

        double chi_sq = 0.0;
        for (Vertex* v : verts) {
            double diff = v->delta_pooled - mu_d;
            chi_sq += diff * diff / v->sigma_sq;
        }
        double tau = std::max(0.0, (chi_sq - (m - 1)) / sum_w);
        tau_sq[d] = tau;
    }

    // For depths with < 3 vertices, borrow from nearest depth
    std::vector<int> depths_with_tau;
    for (auto& [d, _] : tau_sq) depths_with_tau.push_back(d);
    std::sort(depths_with_tau.begin(), depths_with_tau.end());

    if (!depths_with_tau.empty()) {
        for (auto& [d, _] : by_depth) {
            if (tau_sq.find(d) == tau_sq.end()) {
                int nearest = depths_with_tau[0];
                for (int d2 : depths_with_tau) {
                    if (std::abs(d2 - d) < std::abs(nearest - d)) nearest = d2;
                }
                tau_sq[d] = tau_sq[nearest];
                if (mu_delta.find(d) == mu_delta.end()) {
                    auto it = mu_delta.find(nearest);
                    mu_delta[d] = (it != mu_delta.end()) ? it->second : 0.0;
                }
            }
        }
    }

    steps_since_tau_recompute = 0;
}

void DecisionMesh::maybe_recompute_tau_sq() {
    steps_since_tau_recompute++;

    if (steps_since_tau_recompute >= tau_sq_recompute_interval) {
        recompute_tau_sq();
        return;
    }

    // Check if any depth has grown by 50%
    std::map<int, int> by_depth;
    for (Vertex* v : vertices) {
        if (v->parent_edge != nullptr && v->sigma_sq < 1e300) {
            by_depth[v->depth]++;
        }
    }

    for (auto& [d, count] : by_depth) {
        auto it = tau_sq_vertex_counts.find(d);
        int old_count = (it != tau_sq_vertex_counts.end()) ? it->second : 0;
        if (old_count > 0 && count >= (int)(old_count * 1.5)) {
            recompute_tau_sq();
            return;
        }
    }
}

// --- Prediction: walk tree to leaf face, then affine interpolate ---

double DecisionMesh::predict(double px, double py) const {
    // Walk the binary tree from root to leaf: O(depth) = O(log faces)
    const TreeNode* node = root;
    while (node->has_split) {
        double val = node->split_normal[0] * px + node->split_normal[1] * py;
        node = (val >= node->split_intercept) ? node->p : node->m;
    }

    // node->face is the leaf face containing (px, py)
    Face* f = node->face;
    if (!f) return 0.0;

    Vertex* v0 = f->vertices[0];
    Vertex* v1 = f->vertices[1];
    Vertex* v2 = f->vertices[2];
    double h0 = v0->height, h1 = v1->height, h2 = v2->height;

    // Compute barycentric coordinates for the query point
    double ax = v1->x - v0->x, ay = v1->y - v0->y;
    double bx = v2->x - v0->x, by = v2->y - v0->y;
    double det = ax * by - ay * bx;

    if (std::abs(det) < 1e-14) {
        // Degenerate triangle: return average height
        return (h0 + h1 + h2) / 3.0;
    }

    double inv = 1.0 / det;
    double dpx = px - v0->x, dpy = py - v0->y;
    double u = (dpx * by - dpy * bx) * inv;  // weight for v1
    double v = (ax * dpy - ay * dpx) * inv;   // weight for v2
    double w = 1.0 - u - v;                   // weight for v0

    // Clamp barycentric coords to [0,1] to avoid extrapolation on thin triangles
    // when the query point falls slightly outside due to numerical issues
    double cw = std::max(0.0, std::min(1.0, w));
    double cu = std::max(0.0, std::min(1.0, u));
    double cv = std::max(0.0, std::min(1.0, v));
    double total = cw + cu + cv;
    if (total > 0) {
        cw /= total; cu /= total; cv /= total;
    } else {
        cw = cu = cv = 1.0 / 3.0;
    }

    return cw * h0 + cu * h1 + cv * h2;
}

std::vector<double> DecisionMesh::predict_batch(const std::vector<double>& px,
                                                 const std::vector<double>& py) const {
    int n = (int)px.size();
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = predict(px[i], py[i]);
    }
    return result;
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
