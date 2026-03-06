#include "vertex.h"
#include "edge.h"
#include "face.h"
#include "mesh.h"
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>

extern CascadeCounters g_counters;

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
        depth = std::max(parent_edge->vertex0->depth, parent_edge->vertex1->depth) + 1;
        // Register with parent edge endpoints for prior propagation
        if (real_vertex) {
            parent_edge->vertex0->prior_children.insert(this);
            parent_edge->vertex1->prior_children.insert(this);
        }
    } else {
        height = 0.0;
        depth = 0;
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
    g_counters.cascade_activations++;
    active = true;
    height = new_height;

    // Set EB posterior fields before split (new midpoints need these during split)
    if (mesh->use_eb && parent_edge != nullptr) {
        delta_pooled = height - mu_lin();
        double inv_sigma = (sigma_sq < 1e300) ? (1.0 / sigma_sq) : 0.0;
        double xTx_reg = inv_sigma + lambda_v;
        sigma_pooled = (xTx_reg > 0) ? (1.0 / xTx_reg) : 1e300;
    }

    parent_edge->split();

    std::set<Vertex*> to_update(affected_vertices);
    if (mesh->use_eb) {
        to_update.insert(prior_children.begin(), prior_children.end());
    }
    for (Vertex* v : to_update) {
        v->update_info();
    }
    loss_reduction = 0;
    mesh->heap_set(this, 0);
}

void Vertex::update_info() {
    g_counters.update_info_calls++;
    auto result = loc_regress();
    affected_vertices = result.affected;
    affected_points = result.n_points;

    double xTx = result.xTx;
    double xTr = result.xTr;
    double rTr = result.rTr;

    sigma_sq = (xTx > 0) ? (1.0 / xTx) : 1e300;

    if (mesh->use_eb) {
        auto [lv, mu_v] = compute_eb_params(xTx);
        lambda_v = lv;

        if (lv > 0 && xTx > 0) {
            double xTx_reg = xTx + lv;
            double xTr_reg = xTr + lv * mu_v;
            double beta_reg = xTr_reg / xTx_reg;

            double beta_orig = height;
            double orig_loss = beta_orig * beta_orig * xTx
                             - 2.0 * beta_orig * xTr + rTr
                             + lv * (beta_orig - mu_v) * (beta_orig - mu_v);
            double sse_post = rTr + lv * mu_v * mu_v - xTr_reg * xTr_reg / xTx_reg;

            new_height = beta_reg;
            loss_reduction = std::max(0.0, orig_loss - sse_post);
            sigma_pooled = 1.0 / xTx_reg;
            delta_pooled = beta_reg - mu_lin();
        } else {
            new_height = result.beta_opt;
            loss_reduction = result.loss_reduction;
            if (parent_edge != nullptr && xTx > 0) {
                delta_pooled = result.beta_opt - mu_lin();
                sigma_pooled = sigma_sq;
            }
        }
    } else {
        new_height = result.beta_opt;
        loss_reduction = result.loss_reduction;
    }

    if (!disqualified) {
        mesh->heap_set(this, -loss_reduction);
    }
}

void Vertex::update_height() {
    height = new_height;
    if (parent_edge != nullptr) {
        delta_pooled = height - mu_lin();
    }
    std::set<Vertex*> to_update(affected_vertices);
    if (mesh->use_eb) {
        to_update.insert(prior_children.begin(), prior_children.end());
    }
    for (Vertex* v : to_update) {
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
    res.xTx = 0.0;
    res.xTr = 0.0;
    res.rTr = 0.0;

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

    double xTx = 0.0, xTr = 0.0, rTr = 0.0;
    int total_points = 0;

    for (Face* f : fr.faces) {
        int si = -1;
        for (int i = 0; i < 3; ++i) {
            if (f->vertices[i] == this) { si = i; break; }
        }
        if (si < 0 || f->n_covered == 0) continue;

        total_points += f->n_covered;

        double h[3];
        for (int i = 0; i < 3; ++i)
            h[i] = f->vertices[i]->height;

        xTx += f->S_ww[si][si];

        double xTr_face = f->S_wy[si];
        for (int j = 0; j < 3; ++j) {
            if (j == si) continue;
            xTr_face -= h[j] * f->S_ww[si][j];
        }
        xTr += xTr_face;

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

    double beta_orig = height;
    double orig_loss = rTr - 2.0 * beta_orig * xTr + beta_orig * beta_orig * xTx;

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
    res.xTx = xTx;
    res.xTr = xTr;
    res.rTr = rTr;
    return res;
}

// --- Empirical Bayes helpers ---

double Vertex::mu_lin() const {
    if (parent_edge == nullptr) return 0.0;
    return (parent_edge->vertex0->height + parent_edge->vertex1->height) / 2.0;
}

double Vertex::compute_overlap_factor(Vertex* a, Vertex* b) const {
    // Collect faces of a
    std::set<Face*> faces_a, faces_b;
    for (Edge* e : a->edges) {
        for (int s = 0; s < 2; ++s) {
            Face* f = e->faces[s];
            if (f) {
                for (int i = 0; i < 3; ++i) {
                    if (f->vertices[i] == a) { faces_a.insert(f); break; }
                }
            }
        }
    }
    for (Edge* e : b->edges) {
        for (int s = 0; s < 2; ++s) {
            Face* f = e->faces[s];
            if (f) {
                for (int i = 0; i < 3; ++i) {
                    if (f->vertices[i] == b) { faces_b.insert(f); break; }
                }
            }
        }
    }

    // shared faces
    int n_shared = 0, n_a = 0, n_b = 0;
    for (Face* f : faces_a) n_a += f->n_covered;
    for (Face* f : faces_b) n_b += f->n_covered;
    for (Face* f : faces_a) {
        if (faces_b.count(f)) n_shared += f->n_covered;
    }

    int denom = n_a + n_b - n_shared;
    if (denom <= 0) return 0.0;
    return (double)n_shared / denom;
}

std::pair<double, double> Vertex::compute_eb_params(double xTx) const {
    if (parent_edge == nullptr || xTx <= 0) return {0.0, 0.0};

    auto it = mesh->tau_sq.find(depth);
    if (it == mesh->tau_sq.end()) return {0.0, 0.0};
    double tau = it->second;

    Vertex* a = parent_edge->vertex0;
    Vertex* b = parent_edge->vertex1;
    double ml = (a->height + b->height) / 2.0;

    // Non-corner parents
    std::vector<Vertex*> non_corner;
    if (a->parent_edge != nullptr) non_corner.push_back(a);
    if (b->parent_edge != nullptr) non_corner.push_back(b);

    double delta_prior;
    if (!non_corner.empty()) {
        delta_prior = 0.0;
        for (Vertex* v : non_corner) delta_prior += v->delta_pooled;
        delta_prior /= non_corner.size();
    } else {
        auto md_it = mesh->mu_delta.find(depth);
        delta_prior = (md_it != mesh->mu_delta.end()) ? md_it->second : 0.0;
    }

    double mu_v = ml + delta_prior;

    double overlap_frac = compute_overlap_factor(a, b);
    double s_v = 1.0 / std::max(1e-15, 1.0 - overlap_frac);

    double sigma_prior;
    std::vector<double> parent_sigmas;
    for (Vertex* v : non_corner) {
        if (v->sigma_pooled < 1e300) {
            parent_sigmas.push_back(v->sigma_pooled);
        }
    }
    if (!parent_sigmas.empty()) {
        double avg_sigma = 0.0;
        for (double s : parent_sigmas) avg_sigma += s;
        avg_sigma /= parent_sigmas.size();
        sigma_prior = tau + s_v * avg_sigma;
    } else {
        sigma_prior = tau;
    }

    double lv;
    if (sigma_prior <= 0) {
        lv = 1e12;
    } else {
        lv = 1.0 / sigma_prior;
    }

    return {lv, mu_v};
}
