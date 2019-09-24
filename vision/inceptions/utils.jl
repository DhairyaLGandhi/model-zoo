using Pkg
Pkg.activate(".")

using Flux, Flux.Tracker, Flux.Optimise
using Flux: onehotbatch
using CuArrays
using JuliaDB, Images
using Base.Iterators
using StatsBase: shuffle, sample

train_path = "train"

function get_labels(path = "trainLabels.csv")
  JuliaDB.loadtable(path)
end

function create_labels(t)
  onehotbatch([t[i].level == 0 ? 0 : 1 for i in 1:length(t)], 0:1)
end

getarray(X) = float.(permutedims(channelview(X), (2, 3, 1)))

"""
make_batch(t; batch_size = 12, rsize = (299,299), train_path = train_path)
"""
function make_batch(t; batch_size = 12, rsize = (299,299), train_path = train_path)
  clean = filter(x -> x.level == 0, t)
  dirty =  filter(x -> x.level != 0, t)
  o = merge(clean[sample(1:end, Int64(batch_size/2), replace = false)],
           dirty[sample(1:end, Int64(batch_size/2), replace = false)])
  o = o[shuffle(1:end)]
  bt = Images.load.(joinpath.("train", column(o, :image).*".jpeg"))
  bt = imresize.(bt, rsize...)
  bt = getarray.(bt)
  bt = cat(bt..., dims =4)
  bt = bt |> gpu
  ls = create_labels(o) |> gpu
  (bt,ls)
end
