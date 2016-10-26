/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef KERAS_MODEL_H_
#define KERAS_MODEL_H_

#include <chrono>
#include <math.h>
#include <string>
#include <vector>

#define KASSERT(x, ...) \
    if (!(x)) { \
        printf("KASSERT: %s(%d): ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        return false; \
    }

#define KASSERT_EQ(x, y, eps) \
    if (fabs(x - y) > eps) { printf("KASSERT: Expected %f, got %f\n", y, x); return false; }

#ifdef DEBUG
#define KDEBUG(x, ...) \
    if (!(x)) { \
        printf("%s(%d): ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        exit(-1); \
    }
#else
#define KDEBUG(x, ...) ;
#endif

class Tensor {
public:
    Tensor() {}

    Tensor(int i)
    {
        Resize(i);
    }

    Tensor(int i, int j)
    {
        Resize(i, j);
    }

    Tensor(int i, int j, int k)
    {
        Resize(i, j, k);
    }

    Tensor(int i, int j, int k, int l)
    {
        Resize(i, j, k, l);
    }

    void Resize(int i)
    {
        dims_ = {i};
        data_.resize(i);
    }

    void Resize(int i, int j)
    {
        dims_ = {i, j};
        data_.resize(i * j);
    }

    void Resize(int i, int j, int k)
    {
        dims_ = {i, j, k};
        data_.resize(i * j * k);
    }

    void Resize(int i, int j, int k, int l)
    {
        dims_ = {i, j, k, l};
        data_.resize(i * j * k * l);
    }

    inline void Flatten()
    {
        KDEBUG(dims_.size() > 0, "Invalid tensor");

        int elements = dims_[0];
        for (unsigned int i = 1; i < dims_.size(); i++)
        {
            elements *= dims_[i];
        }
        dims_ = {elements};
    }

    inline float& operator()(int i)
    {
        KDEBUG(dims_.size() == 1, "Invalid indexing for tensor");
        KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);

        return data_[i];
    }

    inline float& operator()(int i, int j)
    {
        KDEBUG(dims_.size() == 2, "Invalid indexing for tensor");
        KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
        KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);

        return data_[dims_[1] * i + j];
    }

    inline float& operator()(int i, int j, int k)
    {
        KDEBUG(dims_.size() == 3, "Invalid indexing for tensor");
        KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
        KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);
        KDEBUG(k < dims_[2] && k >= 0, "Invalid k: %d (max %d)", k, dims_[2]);

        return data_[dims_[2] * (dims_[1] * i + j) + k];
    }

    inline float& operator()(int i, int j, int k, int l)
    {
        KDEBUG(dims_.size() == 4, "Invalid indexing for tensor");
        KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
        KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);
        KDEBUG(k < dims_[2] && k >= 0, "Invalid k: %d (max %d)", k, dims_[2]);
        KDEBUG(l < dims_[3] && l >= 0, "Invalid l: %d (max %d)", l, dims_[3]);

        return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
    }

    void Print()
    {
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

    void PrintShape()
    {
        printf("(");
        for (unsigned int i = 0; i < dims_.size(); i++)
        {
            printf("%d ", dims_[i]);
        }
        printf(")\n");
    }


    std::vector<int> dims_;
    std::vector<float> data_;
};


class KerasLayer {
public:
    KerasLayer() {}

    virtual ~KerasLayer() {}

    virtual bool LoadLayer(std::ifstream* file) = 0;

    virtual bool Apply(Tensor* in, Tensor* out) = 0;
};

class KerasLayerActivation : public KerasLayer {
public:
    enum ActivationType
    {
        kLinear = 1,
        kRelu = 2,
    };

    KerasLayerActivation()
    : activation_type_(ActivationType::kLinear)
    {}

    virtual ~KerasLayerActivation() {}

    virtual bool LoadLayer(std::ifstream* file);

    virtual bool Apply(Tensor* in, Tensor* out);

private:

    ActivationType activation_type_;
};

class KerasLayerDense : public KerasLayer {
public:
    KerasLayerDense() {}

    virtual ~KerasLayerDense() {}

    virtual bool LoadLayer(std::ifstream* file);

    virtual bool Apply(Tensor* in, Tensor* out);

private:

    Tensor weights_;
    Tensor biases_;

    KerasLayerActivation activation_;
};

class KerasLayerConvolution2d : public KerasLayer {
public:
    KerasLayerConvolution2d() {}

    virtual ~KerasLayerConvolution2d() {}

    virtual bool LoadLayer(std::ifstream* file);

    virtual bool Apply(Tensor* in, Tensor* out);

private:

    Tensor weights_;
    Tensor biases_;
};

class KerasLayerFlatten : public KerasLayer {
public:
    KerasLayerFlatten() {}

    virtual ~KerasLayerFlatten() {}

    virtual bool LoadLayer(std::ifstream* file);

    virtual bool Apply(Tensor* in, Tensor* out);

private:

};

class KerasLayerElu : public KerasLayer {
public:
    KerasLayerElu()
    : alpha_(1.0f)
    {}

    virtual ~KerasLayerElu() {}

    virtual bool LoadLayer(std::ifstream* file);

    virtual bool Apply(Tensor* in, Tensor* out);

private:

    float alpha_;
};

class KerasModel {
public:

    enum LayerType
    {
        kDense = 1,
        kConvolution2d = 2,
        kFlatten = 3,
        kElu = 4,
        kActivation = 5
    };

    KerasModel()
    {}

    ~KerasModel()
    {
        for (unsigned int i = 0; i < layers_.size(); i++)
        {
            delete layers_[i];
        }
    }

    bool LoadModel(const std::string& filename);

    bool Apply(Tensor* in, Tensor* out);

private:
    std::vector<KerasLayer *> layers_;
};

class KerasTimer {
public:
    KerasTimer() {}

    void Start()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double Stop()
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> now = 
            std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = now - start_;

        return diff.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#endif // KERAS_MODEL_H_
