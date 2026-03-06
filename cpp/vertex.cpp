#include "vertex.h"
#include "edge.h"
#include "face.h"
#include "mesh.h"
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>

int Vertex::_seq = 0;

Vertex::Vertex(DecisionMesh* mesh, double x, double y, bool active,
               Edge* parent_edge, bool real_vertex)
    : _id(Vertex::_seq++), x(x), y(y), active(active),
      real_vertex(real_vertex), mesh(mesh), parent_edge(parent_edge)
{
    if (real_vertex) {
        mesh->vertices.insert(this);
    }
    if (parent_edge) {
        height = (parent_edge->vertex0->height + parent_edge->vertex1->height) / 2.0;
    } else {
        height = 0.0;
    }
}

std::string Vertex::sid() const {
    char buf[16];
    snprintf(buf, sizeof(buf), "V%03d", _id);
    return buf;
}

void Vertex::add_edge(Edge* e) {
    edges.insert(e);
    Vertex* other = e->other_vertex(this);
    if (other) neighbors.insert(other);
}

void Vertex::remove_edge(Edge* e) {
    edges.erase(e);
    Vertex* other = e->other_vertex(this);
    if (other) neighbors.erase(other);
}

void Vertex::activate() {
    if (active) return;
    active = true;
    height = new_height;
    parent_edge->split();
    for (Vertex* v : affected_vertices) {
        v->update_info();
    }
    loss_reduction = 0;
    mesh->heap_set(this, 0);
}

void Vertex::update_info() {
    auto result = loc_regress();
    new_height = result.beta_opt;
    loss_reduction = result.loss_reduction;
    affected_vertices = result.affected;
    affected_points = result.n_points;
    if (!disqualified) {
        mesh->heap_set(this, -loss_reduction);
    }
}

void Vertex::update_height() {
    height = new_height;
    for (Vertex* v : affected_vertices) {
        v->update_info();
    }
    loss_reduction = 0;
    mesh->heap_set(this, 0);
}

Vertex::FacesResult Vertex::get_faces() const {
    FacesResult result;
    for (Edge* e : edges) {
        for (int s = 0; s < 2; ++s) {
            Face* f = e->faces[s];
            if (f) {
                bool has_self = false;
                for (int i = 0; i < 3; ++i) {
                    if (f->vertices[i] == this) { has_self = true; break; }
                }
                if (has_self) {
                    result.faces.insert(f);
                }
            }
        }
    }
    for (Face* f : result.faces) {
        for (int i = 0; i < 3; ++i) {
            if (f->vertices[i] != this) {
                result.neighbors.insert(f->vertices[i]);
            }
        }
    }
    return result;
}

Vertex::FacesResult Vertex::get_sim_faces() const {
    FacesResult result;
    if (!parent_edge) return result;

    for (int s = 0; s < 2; ++s) {
        Face* face = parent_edge->faces[s];
        if (!face) continue;
        // Find which edge index in face corresponds to parent_edge
        int idx = -1;
        for (int i = 0; i < 3; ++i) {
            if (face->edges[i] == parent_edge) { idx = i; break; }
        }
        if (idx < 0) continue;
        if (face->sub_divisions[idx].plus_face)
            result.faces.insert(face->sub_divisions[idx].plus_face);
        if (face->sub_divisions[idx].minus_face)
            result.faces.insert(face->sub_divisions[idx].minus_face);
    }

    for (Face* f : result.faces) {
        for (int i = 0; i < 3; ++i) {
            if (f->vertices[i] != this) {
                result.neighbors.insert(f->vertices[i]);
            }
        }
    }
    return result;
}

std::set<Vertex*> Vertex::neighbors_after_split() const {
    std::set<Vertex*> nbrs;
    if (!parent_edge) return nbrs;
    nbrs.insert(const_cast<Vertex*>(this));
    nbrs.insert(parent_edge->vertex0);
    nbrs.insert(parent_edge->vertex1);
    if (parent_edge->opposing_vertices[0])
        nbrs.insert(parent_edge->opposing_vertices[0]);
    if (parent_edge->opposing_vertices[1])
        nbrs.insert(parent_edge->opposing_vertices[1]);
    return nbrs;
}

Vertex::LocRegressResult Vertex::loc_regress() {
    LocRegressResult res;
    res.beta_opt = height;
    res.loss_reduction = 0.0;
    res.n_points = 0;

    FacesResult fr;
    if (active) {
        fr = get_faces();
    } else {
        fr = get_sim_faces();
    }
    res.affected = fr.neighbors;

    if (fr.faces.empty()) {
        return res;
    }

    // Collect all vertices from all faces
    std::set<Vertex*> all_verts;
    for (Face* f : fr.faces) {
        for (int i = 0; i < 3; ++i)
            all_verts.insert(f->vertices[i]);
    }

    // Map vertices to column indices
    std::map<Vertex*, int> col_key;
    int col_idx = 0;
    for (Vertex* v : all_verts) {
        col_key[v] = col_idx++;
    }
    int n_cols = (int)all_verts.size();

    // Build mask union and design matrix
    // First pass: figure out which global data rows are covered
    std::vector<bool> covered(mesh->n_points, false);
    for (Face* f : fr.faces) {
        for (int i = 0; i < mesh->n_points; ++i) {
            if (f->mask[i]) covered[i] = true;
        }
    }

    // Count covered rows
    std::vector<int> row_indices;
    for (int i = 0; i < mesh->n_points; ++i) {
        if (covered[i]) row_indices.push_back(i);
    }
    int n_rows = (int)row_indices.size();
    if (n_rows == 0) return res;

    // Map global index -> local row index
    std::map<int, int> row_map;
    for (int r = 0; r < n_rows; ++r) {
        row_map[row_indices[r]] = r;
    }

    // Build design matrix X (n_rows x n_cols), initialized to 0
    std::vector<double> X_mat(n_rows * n_cols, 0.0);

    for (Face* f : fr.faces) {
        for (size_t ci = 0; ci < f->coords_indices.size(); ++ci) {
            int gi = f->coords_indices[ci];
            auto it = row_map.find(gi);
            if (it == row_map.end()) continue;
            int r = it->second;
            for (int vi = 0; vi < 3; ++vi) {
                int c = col_key[f->vertices[vi]];
                X_mat[r * n_cols + c] = f->coords_weights[ci][vi];
            }
        }
    }

    // Build y vector
    std::vector<double> y(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        y[r] = mesh->values[row_indices[r]];
    }

    // Self column index
    int self_col = col_key[this];

    // Correction from neighbors
    std::vector<double> correction(n_rows, 0.0);
    for (auto& [v, c] : col_key) {
        if (v == this) continue;
        if (fr.neighbors.count(v) == 0 && v != this) {
            // This is a vertex from faces but not a neighbor
        }
        double h = v->height;
        for (int r = 0; r < n_rows; ++r) {
            correction[r] += X_mat[r * n_cols + c] * h;
        }
    }

    // Extract self column
    std::vector<double> x_self(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        x_self[r] = X_mat[r * n_cols + self_col];
    }

    // Residual r = y - correction
    std::vector<double> r(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        r[i] = y[i] - correction[i];
    }

    // Original loss with current height
    double beta_orig = height;
    double orig_loss = 0.0;
    for (int i = 0; i < n_rows; ++i) {
        double diff = r[i] - x_self[i] * beta_orig;
        orig_loss += diff * diff;
    }

    // Optimal 1D least squares
    double xTx = 0.0, xTr = 0.0, rTr = 0.0;
    for (int i = 0; i < n_rows; ++i) {
        xTx += x_self[i] * x_self[i];
        xTr += x_self[i] * r[i];
        rTr += r[i] * r[i];
    }

    double beta_opt, sse_post;
    if (xTx > 0.0) {
        beta_opt = xTr / xTx;
        sse_post = rTr - (xTr * xTr) / xTx;
    } else {
        beta_opt = beta_orig;
        sse_post = rTr;
    }

    res.beta_opt = beta_opt;
    res.loss_reduction = orig_loss - sse_post;
    res.n_points = n_rows;
    return res;
}
