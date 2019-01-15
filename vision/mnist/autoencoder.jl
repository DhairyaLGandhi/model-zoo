# Autoencoders on Flux

# Neural networks come in all shapes and sizes, and they are capable of a lot
# more than classification and regression. In this notebook, we will explore
# how neural networks can act as a data storage sink or as an efficient
# compression algorithm, to store all sorts of data.

# Autoencoders perform an unsupervised learning task. Specifically, they try
# to learn an identity function whilst also learning an intermediate representation
# which can encode the input datapoint in a smaller feature set that can be
# used to retrieve our original datapoint. The decoding task may be lossy.

# That's a very basic introduction to what an autoencoder is, and all the
# possibilities it opens up. So with that sorted, let's start coding.

# [Flux.jl](https://github.com/FluxML/Flux.jl) is an excellent package for
# deep learning and more in julia. It is really flexible and simple to 
# hack on, since its julia all the way through. It melts into the language
# semantics and holds up user defined functions (and other packages) very
# well. It also has conveneient dataloaders for the MNIST hand writing
# recognition, so let's start with adding it in our environment.

using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, mse, throttle
using Base.Iterators: partition
using Juno: @progress
# using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.

imgs = MNIST.images()

# Partition into batches of size 1000
data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, 1000)]
data = gpu.(data)

N = 32 # Size of the encoding

# You can try to make the encoder/decoder network larger
# Also, the output of encoder is a coding of the given input.
# In this case, the input dimension is 28^2 and the output dimension of
# encoder is 32. This implies that the coding is a compressed representation.
# We can make lossy compression via this `encoder`.
encoder = Dense(28^2, N, leakyrelu) |> gpu
decoder = Dense(N, 28^2, leakyrelu) |> gpu

m = Chain(encoder, decoder)

loss(x) = mse(m(x), x)

evalcb = throttle(() -> @show(loss(data[1])), 5)
opt = ADAM(params(m))

@epochs 10 Flux.train!(loss, zip(data), opt, cb = evalcb)

# Sample output

using Images

img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

function sample()
  # 20 random digits
  before = [imgs[i] for i in rand(1:length(imgs), 20)]
  # Before and after images
  after = img.(map(x -> cpu(m)(float(vec(x))).data, before))
  # Stack them all together
  hcat(vcat.(before, after)...)
end

cd(@__DIR__)

save("sample.png", sample())
