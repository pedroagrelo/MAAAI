

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles
using JLD2
using Images


function fileNamesFolder(folderName::String, extension::String)
    extension = uppercase(extension)

    files = filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName))
    names = first.(splitext.(files))

    return sort(names)
end






function loadDataset(datasetName::String, datasetFolder::String;
                     datasetType::DataType=Float32)
    
    # Construir la ruta completa del archivo
    filePath = joinpath(datasetFolder, datasetName * ".tsv")
    
    # Verificar que el archivo existe
    if !isfile(filePath)
        return nothing
    end
    
    # Cargar todo el archivo con readdlm usando tabulador como separador
    data = readdlm(filePath, '\t', String)
    
    # La primera fila son los encabezados
    headers = data[1, :]
    
    # Buscar la columna que corresponde al target
    target_col = findfirst(==("target"), headers)
    
    # Todas las filas excepto la primera
    data_values = data[2:end, :]
    
    # Convertir las entradas al tipo deseado y eliminar la columna de target
    inputs = parse.(datasetType, data_values[:, setdiff(1:end, target_col)])
    
    # Tomar la columna target y convertir a booleano
    targets = data_values[:, target_col] .== "1"
    
    # Devolver la tupla (inputs, targets)
    return (inputs, targets)
end



function loadImage(imageName::String, datasetFolder::String;
                   datasetType::DataType=Float32, resolution::Int=128)
    
    # Construir la ruta completa del archivo con extensión .tif
    filePath = joinpath(datasetFolder, imageName * ".tif")
    
    # Verificar existencia
    if !isfile(filePath)
        return nothing
    end
    
    # Cargar la imagen
    img = load(filePath)
    
    # Convertir a escala de grises
    img_gray = gray.(img)
    
    # Cambiar resolución
    img_resized = imresize(img_gray, (resolution, resolution))
    
    # Convertir tipo de datos
    img_typed = convert.(datasetType, img_resized)
    
    return img_typed
end



function convertImagesNCHW(imageVector::Vector{<:AbstractArray{<:Real,2}})
    imagesNCHW = Array{eltype(imageVector[1]), 4}(undef, length(imageVector), 1, size(imageVector[1],1), size(imageVector[1],2));
    for numImage in Base.OneTo(length(imageVector))
        imagesNCHW[numImage,1,:,:] .= imageVector[numImage];
    end;
    return imagesNCHW;
end;


function loadImagesNCHW(datasetFolder::String;
                         datasetType::DataType=Float32,
                         resolution::Int=128)
    
    # Obtener todos los nombres de archivos .tif en la carpeta
    imageNames = fileNamesFolder(datasetFolder, "tif")
    
    # Cargar todas las imágenes con broadcast de loadImage
    images = loadImage.(imageNames, Ref(datasetFolder);
                        datasetType=datasetType,
                        resolution=resolution)
    
    # Filtrar posibles `nothing` si algún archivo no se cargó
    images = filter(!isnothing, images)
    
    # Convertir el vector de imágenes a formato NCHW
    imagesNCHW = convertImagesNCHW(images)
    
    return imagesNCHW
end



showImage(image      ::AbstractArray{<:Real,2}                                      ) = display(Gray.(image));
showImage(imagesNCHW ::AbstractArray{<:Real,4}                                      ) = display(Gray.(     hcat([imagesNCHW[ i,1,:,:] for i in 1:size(imagesNCHW ,1)]...)));
showImage(imagesNCHW1::AbstractArray{<:Real,4}, imagesNCHW2::AbstractArray{<:Real,4}) = display(Gray.(vcat(hcat([imagesNCHW1[i,1,:,:] for i in 1:size(imagesNCHW1,1)]...), hcat([imagesNCHW2[i,1,:,:] for i in 1:size(imagesNCHW2,1)]...))));




function loadMNISTDataset(datasetFolder::String; 
                          labels::AbstractArray{Int,1}=0:9, 
                          datasetType::DataType=Float32)

    # Ruta del archivo
    filePath = joinpath(datasetFolder, "MNIST.jld2")
    if !isfile(filePath)
        return nothing
    end

    # Abrir archivo y extraer las variables
    trainImages = nothing
    trainTargets = nothing
    testImages = nothing
    testTargets = nothing

    jldopen(filePath, "r") do file
        all_keys = collect(keys(file))  # Obtener todas las claves
         # Convertir cada imagen individualmente
        trainImages = [datasetType.(img) for img in read(file, all_keys[1])]
        trainTargets = read(file, all_keys[2])
        testImages = [datasetType.(img) for img in read(file, all_keys[3])]
        testTargets = read(file, all_keys[4])
    end

    # Filtrado de etiquetas
    if -1 in labels
        trainTargets[.!in.(trainTargets, [setdiff(labels,-1)])] .= -1
        testTargets[.!in.(testTargets, [setdiff(labels,-1)])] .= -1
        trainIndices = trues(length(trainTargets))
        testIndices = trues(length(testTargets))
    else
        trainIndices = in.(trainTargets, [labels])
        testIndices = in.(testTargets, [labels])
    end

    # Filtrar imágenes y targets
    trainImagesFiltered = trainImages[trainIndices]
    trainTargetsFiltered = trainTargets[trainIndices]
    testImagesFiltered = testImages[testIndices]
    testTargetsFiltered = testTargets[testIndices]

    # Convertir a NCHW
    trainImagesNCHW = convertImagesNCHW(trainImagesFiltered)
    testImagesNCHW = convertImagesNCHW(testImagesFiltered)

    return (trainImagesNCHW, trainTargetsFiltered, testImagesNCHW, testTargetsFiltered)
end



function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    # Ordenar los datos
    uniqueData = sort(unique(data));
    # Obtener diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    # Tomar la diferencia menor
    minDifference = differences[1];
    # Si todas las diferencias son multiplos exactos (valores enteros) de esa diferencia, entonces es un vector de valores discretos
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
end


function cyclicalEncoding(data::AbstractArray{<:Real,1})
    # Calcular el intervalo discreto m
    m = intervalDiscreteVector(data)
    
    # Rango de los datos
    minD = minimum(data)
    maxD = maximum(data)
    
    # Evitar división por cero si todos los datos son iguales
    rangeD = maxD - minD
    denom = rangeD + m > 0 ? rangeD + m : 1.0
    
    # Calcular los ángulos
    angles = 2π .* (data .- minD) ./ denom
    
    # Codificación cíclica: senos y cosenos
    sinValues = sin.(angles)
    cosValues = cos.(angles)
    
    return (sinValues, cosValues)
end




function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float32)
    # Rutas de los archivos
    dataPath = joinpath(datasetFolder, "elec2_data.dat")
    labelPath = joinpath(datasetFolder, "elec2_label.dat")
    
    # Verificar que existen
    if !(isfile(dataPath) && isfile(labelPath))
        return nothing
    end
    
    # Cargar datos y etiquetas
    dataRaw = readdlm(dataPath)
    labelsRaw = vec(Bool.(readdlm(labelPath)))  # Convertir a booleano y vector
    
    # Eliminar columnas 1 (date) y 4 (nswprice)
    dataProcessed = dataRaw[:, setdiff(1:size(dataRaw,2), [1,4])]
    
    # Tomar la columna del día (primera columna de dataProcessed)
    dayColumn = dataProcessed[:,1]
    
    # Codificación cíclica
    sinVals, cosVals = cyclicalEncoding(dayColumn)
    
    # Eliminar la columna del día de dataProcessed
    dataWithoutDay = dataProcessed[:,2:end]
    
    # Concatenar senos y cosenos como primeras columnas
    inputs = hcat(sinVals, cosVals, datasetType.(dataWithoutDay))
    
    return (inputs, labelsRaw)
end




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Flux

indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)
    #
    # Codigo a desarrollar
    #
end;

function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    #
    # Codigo a desarrollar
    #
end;


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    #
    # Codigo a desarrollar
    #
end;

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    #
    # Codigo a desarrollar
    #
end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    #
    # Codigo a desarrollar
    #
end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;
function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    #
    # Codigo a desarrollar
    #
end;


function runHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    prev_S = nothing;
    prev_prev_S = nothing;
    while S!=prev_S && S!=prev_prev_S
        prev_prev_S = prev_S;
        prev_S = S;
        S = stepHopfield(ann, S);
    end;
    return S
end;
function runHopfield(ann::HopfieldNet, dataset::AbstractArray{<:Real,2})
    outputs = copy(dataset);
    for i in 1:size(dataset,1)
        outputs[i,:] .= runHopfield(ann, view(dataset, i, :));
    end;
    return outputs;
end;
function runHopfield(ann::HopfieldNet, datasetNCHW::AbstractArray{<:Real,4})
    outputs = runHopfield(ann, reshape(datasetNCHW, size(datasetNCHW,1), size(datasetNCHW,3)*size(datasetNCHW,4)));
    return reshape(outputs, size(datasetNCHW,1), 1, size(datasetNCHW,3), size(datasetNCHW,4));
end;





function addNoise(datasetNCHW::AbstractArray{<:Bool,4}, ratioNoise::Real)
    #
    # Codigo a desarrollar
    #
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    #
    # Codigo a desarrollar
    #
end;

function randomImages(numImages::Int, resolution::Int)
    #
    # Codigo a desarrollar
    #
end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
predict(model, inputs::AbstractArray) = MLJ.predict(model, MLJ.table(inputs));



using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchTargets(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchLength(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function selectInstances(batch::Batch, indices::Any)
    #
    # Codigo a desarrollar
    #
end;

function joinBatches(batch1::Batch, batch2::Batch)
    #
    # Codigo a desarrollar
    #
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    #
    # Codigo a desarrollar
    #
end;

function addBatch!(memory::Batch, newBatch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;

