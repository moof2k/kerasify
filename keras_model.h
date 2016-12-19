/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef KERAS_MODEL_H_
#define KERAS_MODEL_H_

#include <math.h>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define KASSERT(x, ...)                              \
  if (!(x)) {                                        \
    printf("KASSERT: %s(%d): ", __FILE__, __LINE__); \
    printf(__VA_ARGS__);                             \
    printf("\n");                                    \
    return false;                                    \
  }

#define KASSERT_EQ(x, y, eps)                       \
  if (fabs(x - y) > eps) {                          \
    printf("KASSERT: Expected %f, got %f\n", y, x); \
    return false;                                   \
  }

#ifdef DEBUG
#define KDEBUG(x, ...)                      \
  if (!(x)) {                               \
    printf("%s(%d): ", __FILE__, __LINE__); \
    printf(__VA_ARGS__);                    \
    printf("\n");                           \
    exit(-1);                               \
  }
#else
#define KDEBUG(x, ...) ;
#endif

namespace kerasify {

class Tensor {
 public:
  Tensor() {}

  Tensor(const std::vector<int>& dims, const std::vector<float>& data)
      : dims_(dims), data_(data) {
    KDEBUG(!dims.empty(), "Invalid dimensions");
  }

  Tensor(int i) { Resize(i); }

  Tensor(int i, int j) { Resize(i, j); }

  Tensor(int i, int j, int k) { Resize(i, j, k); }

  Tensor(int i, int j, int k, int l) { Resize(i, j, k, l); }

  void Resize(int i) {
    dims_ = {i};
    data_.resize(i);
  }

  void Resize(int i, int j) {
    dims_ = {i, j};
    data_.resize(i * j);
  }

  void Resize(int i, int j, int k) {
    dims_ = {i, j, k};
    data_.resize(i * j * k);
  }

  void Resize(int i, int j, int k, int l) {
    dims_ = {i, j, k, l};
    data_.resize(i * j * k * l);
  }

  const std::vector<int>& dims() const { return dims_; }

  // Concatenate along first dimension.
  inline bool Append(const Tensor& other) {
    // Check for compatible dimensionality.
    if (dims_.size() != other.dims_.size()) {
      return false;
    }

    // Skip the batch first dimension. All other dimensions should match.
    for (unsigned int i = 1; i < dims_.size(); i++) {
      if (dims_[i] != other.dims_[i]) {
        return false;
      }
    }

    // Concatenate.
    const unsigned int initial_data_size = data_.size();
    dims_[0] += other.dims_[0];  // Update dimensions.
    data_.resize(initial_data_size + other.data_.size());

    unsigned int i = initial_data_size;
    for (const auto value : other.data_) {
      data_[i] = value;
      i++;
    }
    return true;
  }

  inline void Flatten() {
    KDEBUG(dims_.size() > 0, "Invalid tensor");

    int elements = dims_[0];
    for (unsigned int i = 1; i < dims_.size(); i++) {
      elements *= dims_[i];
    }
    dims_ = {elements};
  }

  inline float& operator()(int i) {
    KDEBUG(dims_.size() == 1, "Invalid indexing for tensor");
    KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);

    return data_[i];
  }

  inline float& operator()(int i, int j) {
    KDEBUG(dims_.size() == 2, "Invalid indexing for tensor");
    KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
    KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);

    return data_[dims_[1] * i + j];
  }

  inline float& operator()(int i, int j, int k) {
    KDEBUG(dims_.size() == 3, "Invalid indexing for tensor");
    KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
    KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);
    KDEBUG(k < dims_[2] && k >= 0, "Invalid k: %d (max %d)", k, dims_[2]);

    return data_[dims_[2] * (dims_[1] * i + j) + k];
  }

  inline float& operator()(int i, int j, int k, int l) {
    KDEBUG(dims_.size() == 4, "Invalid indexing for tensor");
    KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
    KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);
    KDEBUG(k < dims_[2] && k >= 0, "Invalid k: %d (max %d)", k, dims_[2]);
    KDEBUG(l < dims_[3] && l >= 0, "Invalid l: %d (max %d)", l, dims_[3]);

    return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
  }

  void Print() {
    if (dims_.size() == 1) {
      printf("[ ");
      for (int i = 0; i < dims_[0]; i++) {
        printf("%f ", (*this)(i));
      }
      printf("]\n");
    } else if (dims_.size() == 2) {
      printf("[\n");
      for (int i = 0; i < dims_[0]; i++) {
        printf(" [ ");
        for (int j = 0; j < dims_[1]; j++) {
          printf("%f ", (*this)(i, j));
        }
        printf("]\n");
      }
      printf("]\n");
    } else if (dims_.size() == 3) {
      printf("[\n");
      for (int i = 0; i < dims_[0]; i++) {
        printf(" [\n");
        for (int j = 0; j < dims_[1]; j++) {
          printf("  [ ");
          for (int k = 0; k < dims_[2]; k++) {
            printf("%f ", (*this)(i, j, k));
          }
          printf("  ]\n");
        }
        printf(" ]\n");
      }
      printf("]\n");
    } else if (dims_.size() == 4) {
      printf("[\n");
      for (int i = 0; i < dims_[0]; i++) {
        printf(" [\n");
        for (int j = 0; j < dims_[1]; j++) {
          printf("  [\n");
          for (int k = 0; k < dims_[2]; k++) {
            printf("   [");
            for (int l = 0; l < dims_[3]; l++) {
              printf("%f ", (*this)(i, j, k, l));
            }
            printf("]\n");
          }
          printf("  ]\n");
        }
        printf(" ]\n");
      }
      printf("]\n");
    }
  }

  void PrintShape() {
    printf("(");
    for (unsigned int i = 0; i < dims_.size(); i++) {
      printf("%d ", dims_[i]);
    }
    printf(")\n");
  }

  std::vector<int> dims_;
  std::vector<float> data_;
};
using TensorMap = std::unordered_map<std::string, Tensor*>;

class KerasLayer {
 public:
  explicit KerasLayer(const std::string& name,
                      const std::vector<std::string>& inbound_layer_names)
      : name_(name), inbound_layer_names_(inbound_layer_names) {}

  virtual ~KerasLayer() = default;

  virtual bool LoadLayer(std::ifstream* file) = 0;

  virtual bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) = 0;

  const std::string& name() const { return name_; }

  const std::vector<std::string>& inbound_layer_names() const {
    return inbound_layer_names_;
  }

 protected:
  const std::string name_;
  const std::vector<std::string> inbound_layer_names_;
};

using KerasLayerMap = std::unordered_map<std::string, KerasLayer*>;

class KerasLayerInput : public KerasLayer {
 public:
  explicit KerasLayerInput(const std::string& name,
                           const std::vector<std::string>& inbound_layer_names)
      : KerasLayer(name, inbound_layer_names) {}

  virtual ~KerasLayerInput() = default;

  bool LoadLayer(std::ifstream* file) override;

  bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) override;
};

class KerasLayerMerge : public KerasLayer {
 public:
  explicit KerasLayerMerge(const std::string& name,
                           const std::vector<std::string>& inbound_layer_names)
      : KerasLayer(name, inbound_layer_names) {}

  virtual ~KerasLayerMerge() = default;

  bool LoadLayer(std::ifstream* file) override;

  bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) override;
};

class KerasLayerActivation : public KerasLayer {
 public:
  enum ActivationType { kLinear = 1, kRelu = 2, kSoftPlus = 3 };

  KerasLayerActivation() : KerasLayerActivation("", {}) {}

  explicit KerasLayerActivation(
      const std::string& name,
      const std::vector<std::string>& inbound_layer_names)
      : KerasLayer(name, inbound_layer_names),
        activation_type_(ActivationType::kLinear) {}

  virtual ~KerasLayerActivation() = default;

  bool LoadLayer(std::ifstream* file) override;

  bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) override;

 private:
  ActivationType activation_type_;
};

class KerasLayerDense : public KerasLayer {
 public:
  explicit KerasLayerDense(const std::string& name,
                           const std::vector<std::string>& inbound_layer_names)
      : KerasLayer(name, inbound_layer_names) {}

  virtual ~KerasLayerDense() = default;

  bool LoadLayer(std::ifstream* file) override;

  bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) override;

 private:
  Tensor weights_;
  Tensor biases_;

  KerasLayerActivation activation_;
};

class KerasLayerConvolution2d : public KerasLayer {
 public:
  explicit KerasLayerConvolution2d(
      const std::string& name,
      const std::vector<std::string>& inbound_layer_names)
      : KerasLayer(name, inbound_layer_names) {}

  virtual ~KerasLayerConvolution2d() = default;

  bool LoadLayer(std::ifstream* file) override;

  bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) override;

 private:
  Tensor weights_;
  Tensor biases_;

  KerasLayerActivation activation_;
};

class KerasLayerFlatten : public KerasLayer {
 public:
  explicit KerasLayerFlatten(
      const std::string& name,
      const std::vector<std::string>& inbound_layer_names)
      : KerasLayer(name, inbound_layer_names) {}

  virtual ~KerasLayerFlatten() = default;

  bool LoadLayer(std::ifstream* file) override;

  bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) override;

 private:
};

class KerasLayerElu : public KerasLayer {
 public:
  explicit KerasLayerElu(const std::string& name,
                         const std::vector<std::string>& inbound_layer_names)
      : KerasLayer(name, inbound_layer_names), alpha_(1.0f) {}

  virtual ~KerasLayerElu() = default;

  bool LoadLayer(std::ifstream* file) override;

  bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) override;

 private:
  float alpha_;
};

class KerasLayerMaxPooling2d : public KerasLayer {
 public:
  explicit KerasLayerMaxPooling2d(
      const std::string& name,
      const std::vector<std::string>& inbound_layer_names)
      : KerasLayer(name, inbound_layer_names),
        pool_size_j_(0),
        pool_size_k_(0) {}

  virtual ~KerasLayerMaxPooling2d() = default;

  bool LoadLayer(std::ifstream* file) override;

  bool Apply(const std::vector<Tensor*>& in_list, Tensor* out) override;

 private:
  unsigned int pool_size_j_;
  unsigned int pool_size_k_;
};

// Represents the graph of layer evaluations needed to materialize one or more
// output layers from one or more inputs to the model.
class KerasGraph {
 public:
  class KerasNode {
   public:
    explicit KerasNode(KerasLayer* layer) : layer_(layer) {}

    bool Initialize(KerasGraph* graph);
    bool Compute();

    void SetResult(const Tensor& in) {
      result_.reset(new Tensor());
      *result_ = in;
    }

    bool Clear() {
      result_.reset(nullptr);
      return true;
    }

    const std::string& name() const { return layer_->name(); }
    Tensor* result() const { return result_.get(); }

   private:
    KerasLayer* layer_;
    std::vector<KerasNode*> inbound_nodes_;
    std::unique_ptr<Tensor> result_;
  };

  KerasGraph() = default;

  bool Initialize(const std::vector<KerasLayer*>& layers);

  bool Evaluate(TensorMap& in_map, TensorMap* out_map);

 protected:
  KerasGraph::KerasNode* GetOrCreateNode(const std::string& layer_name);

 private:
  KerasLayerMap layer_map_;
  std::unordered_map<std::string, std::unique_ptr<KerasNode>> node_map_;
};

class KerasModel {
 public:
  enum LayerType {
    kDense = 1,
    kConvolution2d = 2,
    kFlatten = 3,
    kElu = 4,
    kActivation = 5,
    kMaxPooling2D = 6,
    kInput = 7,
    kMerge = 8
  };

  KerasModel() = default;

  ~KerasModel() {
    for (unsigned int i = 0; i < layers_.size(); i++) {
      delete layers_[i];
    }
  }

  bool LoadModel(const std::string& filename);

  bool Apply(Tensor* in, Tensor* out);

  bool Apply(TensorMap& in_map, TensorMap* out_map);

 private:
  std::vector<KerasLayer*> layers_;
  std::vector<std::string> input_layer_names_;
  std::vector<std::string> output_layer_names_;

  KerasGraph graph_;
};

class KerasTimer {
 public:
  KerasTimer() {}

  void Start() { start_ = std::chrono::high_resolution_clock::now(); }

  double Stop() {
    std::chrono::time_point<std::chrono::high_resolution_clock> now =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = now - start_;

    return diff.count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

}  // namespace kerasify

#endif  // KERAS_MODEL_H_
