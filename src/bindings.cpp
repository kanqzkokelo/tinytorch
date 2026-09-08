/* Pybind11 bindings for tinytorch C/autograd library with shared_ptr lifecycle */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include "tensor.h"
#include "autograd.h"

namespace py = pybind11;

struct TensorDeleter {
    void operator()(Tensor *t) const { if (t) tt_release(t); }
};

struct NodeDeleter {
    void operator()(AGNode *n) const { if (n) ag_release(n); }
};

using SharedTensor = std::shared_ptr<Tensor>;
using SharedNode = std::shared_ptr<AGNode>;

static SharedTensor make_shared_tensor(Tensor *t) {
    if (!t) throw py::value_error("tinytorch: invalid arguments or shape mismatch");
    return SharedTensor(t, TensorDeleter());
}

static SharedNode make_shared_node(AGNode *n) {
    if (!n) throw py::value_error("tinytorch: invalid arguments or shape mismatch");
    return SharedNode(n, NodeDeleter());
}

static py::array_t<float> tensor_to_numpy(Tensor *t) {
    if (!t) return py::array_t<float>();
    int ndim = tt_ndim(t);
    std::vector<long> shp(ndim);
    tt_shape(t, shp.data());
    std::vector<ssize_t> shape(ndim);
    for (int i = 0; i < ndim; i++) shape[i] = (ssize_t)shp[i];
    py::array_t<float> arr(shape);
    size_t bytes = sizeof(float) * (size_t)tt_numel(t);
    memcpy(arr.mutable_data(), tt_data(t), bytes);
    return arr;
}

static SharedTensor tensor_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info info = arr.request();
    int ndim = (int)info.ndim;
    std::vector<long> shape(ndim);
    for (int i = 0; i < ndim; i++) shape[i] = (long)info.shape[i];
    Tensor *t = tt_fromdata(static_cast<float*>(info.ptr), shape.data(), ndim);
    return make_shared_tensor(t);
}

class PyNode {
public:
    SharedNode node;

    PyNode(SharedNode n) : node(n) {}

    static PyNode leaf(py::array_t<float, py::array::c_style | py::array::forcecast> arr, bool req_grad = true) {
        SharedTensor t = tensor_from_numpy(arr);
        AGNode *n = ag_leaf(t.get(), req_grad ? 1 : 0);
        return PyNode(make_shared_node(n));
    }

    py::array_t<float> value() const {
        if (!node) return py::array_t<float>();
        Tensor *v = ag_value(node.get());
        return tensor_to_numpy(v);
    }

    py::object grad() const {
        if (!node) return py::none();
        Tensor *g = ag_grad(node.get());
        if (!g) return py::none();
        return tensor_to_numpy(g);
    }

    void backward(py::object seed = py::none()) {
        if (!node) return;
        if (seed.is_none()) {
            ag_backward(node.get());
        } else {
            py::array_t<float, py::array::c_style | py::array::forcecast> arr = seed.cast<py::array_t<float>>();
            py::buffer_info info = arr.request();
            Tensor *v = ag_value(node.get());
            long need = v ? tt_numel(v) : -1;
            if ((size_t)info.size != (size_t)need)
                throw py::value_error("tinytorch: seed length mismatch");
            ag_backward_from(node.get(), static_cast<float*>(info.ptr), (size_t)info.size);
        }
    }

    PyNode add(const PyNode &other) { return PyNode(make_shared_node(ag_add(node.get(), other.node.get()))); }
    PyNode mul(const PyNode &other) { return PyNode(make_shared_node(ag_mul(node.get(), other.node.get()))); }
    PyNode addscalar(float s) { return PyNode(make_shared_node(ag_addscalar(node.get(), s))); }
    PyNode mulscalar(float s) { return PyNode(make_shared_node(ag_mulscalar(node.get(), s))); }
    PyNode matmul(const PyNode &other) { return PyNode(make_shared_node(ag_matmul(node.get(), other.node.get()))); }
    PyNode relu() { return PyNode(make_shared_node(ag_relu(node.get()))); }
    PyNode softmax() { return PyNode(make_shared_node(ag_softmax(node.get()))); }
    PyNode padones() { return PyNode(make_shared_node(ag_padones(node.get()))); }

    PyNode reshape(std::vector<long> shape) {
        return PyNode(make_shared_node(ag_reshape(node.get(), shape.data(), (int)shape.size())));
    }

    PyNode conv2d(const PyNode &w, py::object b_obj = py::none(), int sh = 1, int sw = 1, int ph = 0, int pw = 0) {
        AGNode *b_ptr = nullptr;
        if (!b_obj.is_none()) {
            PyNode b = b_obj.cast<PyNode>();
            b_ptr = b.node.get();
        }
        return PyNode(make_shared_node(ag_conv2d(node.get(), w.node.get(), b_ptr, sh, sw, ph, pw)));
    }

    PyNode maxpool2d(int ph = 2, int pw = 2, int sh = 2, int sw = 2) {
        return PyNode(make_shared_node(ag_maxpool2d(node.get(), ph, pw, sh, sw)));
    }

    PyNode avgpool2d(int ph = 2, int pw = 2, int sh = 2, int sw = 2) {
        return PyNode(make_shared_node(ag_avgpool2d(node.get(), ph, pw, sh, sw)));
    }
};

PYBIND11_MODULE(tinytorch_pybind, m) {
    m.doc() = "tinytorch C/autograd pybind11 module";

    py::class_<PyNode>(m, "Node")
        .def_static("leaf", &PyNode::leaf, py::arg("arr"), py::arg("requires_grad") = true)
        .def("value", &PyNode::value)
        .def("grad", &PyNode::grad)
        .def("backward", &PyNode::backward, py::arg("seed") = py::none())
        .def("add", &PyNode::add)
        .def("mul", &PyNode::mul)
        .def("addscalar", &PyNode::addscalar)
        .def("mulscalar", &PyNode::mulscalar)
        .def("matmul", &PyNode::matmul)
        .def("relu", &PyNode::relu)
        .def("softmax", &PyNode::softmax)
        .def("padones", &PyNode::padones)
        .def("reshape", &PyNode::reshape)
        .def("conv2d", &PyNode::conv2d, py::arg("w"), py::arg("b") = py::none(),
             py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0)
        .def("maxpool2d", &PyNode::maxpool2d, py::arg("pool_h") = 2, py::arg("pool_w") = 2,
             py::arg("stride_h") = 2, py::arg("stride_w") = 2)
        .def("avgpool2d", &PyNode::avgpool2d, py::arg("pool_h") = 2, py::arg("pool_w") = 2,
             py::arg("stride_h") = 2, py::arg("stride_w") = 2)
        .def("__add__", &PyNode::add)
        .def("__mul__", &PyNode::mul);
}
