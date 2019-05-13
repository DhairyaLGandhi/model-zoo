```@meta
EditURL = "https://github.com/TRAVIS_REPO_SLUG/blob/master/"
```

```@example mlp
using Pkg; Pkg.activate("."); Pkg.instantiate();

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated## using CuArrays
```

Classify MNIST digits with a simple multi-layer-perceptron

```@example mlp
imgs = MNIST.images()
```

Stack images into one large batch

```@example mlp
X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()
```

One-hot-encode the labels

```@example mlp
Y = onehotbatch(labels, 0:9) |> gpu

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

dataset = repeated((X, Y), 200) |> gpu
evalcb = () -> @show(loss(X, Y))
opt = ADAM(params(m))

@info "starting training.."
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

accuracy(X, Y)
```

Test set accuracy

```@example mlp
tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

accuracy(tX, tY)
```

