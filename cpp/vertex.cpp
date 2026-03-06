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

    // Use per-face sufficient statistics to compute 1D least squares
    // without iterating over individual data points.
    //
    // For vertex "self" at index si in face f:
    //   residual_r = y_r - Σ_{j≠si} w_jr * h_j
    //   x_self_r   = w_{si,r}
    //
    // xTx += f.S_ww[si][si]
    // xTr += f.S_wy[si] - Σ_{j≠si} h_j * f.S_ww[si][j]
    // rTr += f.S_yy - 2*Σ_{j≠si} h_j*f.S_wy[j]
    //        + Σ_{j≠si} Σ_{k≠si} h_j*h_k*f.S_ww[j][k]

    double xTx = 0.0, xTr = 0.0, rTr = 0.0;
    int total_points = 0;

    for (Face* f : fr.faces) {
        // Find which vertex index in this face is "self"
        int si = -1;
        for (int i = 0; i < 3; ++i) {
            if (f->vertices[i] == this) { si = i; break; }
        }
        if (si < 0 || f->n_covered == 0) continue;

        total_points += f->n_covered;

        // Gather neighbor heights for this face
        double h[3];
        for (int i = 0; i < 3; ++i)
            h[i] = f->vertices[i]->height;

        xTx += f->S_ww[si][si];

        // xTr contribution
        double xTr_face = f->S_wy[si];
        for (int j = 0; j < 3; ++j) {
            if (j == si) continue;
            xTr_face -= h[j] * f->S_ww[si][j];
        }
        xTr += xTr_face;

        // rTr contribution: ||y - Σ_{j≠si} w_j h_j||²
        double rTr_face = f->S_yy;
        for (int j = 0; j < 3; ++j) {
            if (j == si) continue;
            rTr_face -= 2.0 * h[j] * f->S_wy[j];
            for (int k = 0; k < 3; ++k) {
                if (k == si) continue;
                rTr_face += h[j] * h[k] * f->S_ww[j][k];
            }
        }
        rTr += rTr_face;
    }

    if (total_points == 0) return res;

    // Original loss with current height
    double beta_orig = height;
    double orig_loss = rTr - 2.0 * beta_orig * xTr + beta_orig * beta_orig * xTx;

    // Optimal 1D least squares
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
    res.n_points = total_points;
    return res;
}
