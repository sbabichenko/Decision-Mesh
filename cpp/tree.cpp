#include "tree.h"
#include "mesh.h"
#include "edge.h"
#include "face.h"
#include <sstream>

int TreeNode::_seq = 0;

TreeNode::TreeNode(DecisionMesh* mesh, TreeNode* parent, Face* face)
    : _id(TreeNode::_seq++), mesh(mesh), parent(parent), face(face)
{
    if (face) {
        face->addnode(this);
    }
}

std::string TreeNode::sid() const {
    char buf[16];
    snprintf(buf, sizeof(buf), "N%03d", _id);
    return buf;
}

int TreeNode::depth() const {
    int d = 0;
    const TreeNode* cur = parent;
    while (cur) { ++d; cur = cur->parent; }
    return d;
}

std::string TreeNode::path() const {
    std::string bits;
    const TreeNode* node = this;
    while (node->parent) {
        bits += (node->parent->p == node) ? '+' : '-';
        node = node->parent;
    }
    std::string result(bits.rbegin(), bits.rend());
    return result;
}

void TreeNode::split(Edge* edge, Face* plus, Face* minus) {
    has_split = true;
    split_normal[0] = edge->normal[0];
    split_normal[1] = edge->normal[1];
    split_intercept = edge->intercept;

    p = mesh->make_node(this, plus);
    m = mesh->make_node(this, minus);

    mesh->leaves.erase(this);
    mesh->leaves.insert(p);
    mesh->leaves.insert(m);
}

TreeNode* TreeNode::child_for(char sign) const {
    return sign == '+' ? p : m;
}
