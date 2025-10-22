

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
    if numOutputs == 1
        # Clasificación binaria → salida con 1 neurona y activación sigmoide
        return Chain(
            Dense(numInputs, 1, σ)
        )
    else
        # Clasificación multiclase → salida lineal + softmax
        return Chain(
            Dense(numInputs, numOutputs, identity),
            softmax
        )
    end
end

function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)

  # Localizar capas
    idx_out = indexOutputLayer(previousANN)
    outputLayer    = previousANN[idx_out]
    previousLayers = previousANN[1:(idx_out-1)]

    # Número de entradas y salidas de la capa de salida
    numInputsOutputLayer  = size(outputLayer.weight, 2)
    numOutputsOutputLayer = size(outputLayer.weight, 1)

    # Nueva capa oculta (SkipConnection)
    newHidden = SkipConnection(
        Dense(numInputsOutputLayer, 1, transferFunction),
        (mx, x) -> vcat(x, mx)
    )

    # Nueva capa de salida (según el caso de clasificación)
    if idx_out == length(previousANN)   # 2 clases
        newOutput = Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, outputLayer.σ)
        ann = Chain(previousLayers..., newHidden, newOutput)
    else                                # más de 2 clases
        newOutput = Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity)
        ann = Chain(previousLayers..., newHidden, newOutput, softmax)
    end

    # Inicializar pesos de la nueva capa de salida
    newOutput.weight[:, 1:numInputsOutputLayer] .= outputLayer.weight
    newOutput.weight[:, end] .= 0.0
    newOutput.bias .= outputLayer.bias

    return ann

end;

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5) 
    X, Y = trainingDataset
    @show size(X)
    @show size(Y)
    @show size(ann(X))
    println("---- Depuración dentro de trainClassANN! ----")
    println("size(X): ", size(X))
    println("size(Y): ", size(Y))
    println("size(ann(X)): ", size(ann(X)))
    println("Número de salidas de la red: ", size(ann(X), 1))
    println("--------------------------------------------")

    # --- función de pérdida ---
    #loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)
    loss(m, x, y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(m(x), y) : Flux.Losses.crossentropy(m(x), y)


    # --- optimizador ---
    opt_state = Flux.setup(Adam(learningRate), ann)

    # congelar capas si procede
    if trainOnly2LastLayers
        Flux.freeze!(opt_state.layers[1:(indexOutputLayer(ann)-2)])
    end

    # --- historial ---
    trainingLosses = Float32[]
    push!(trainingLosses, loss(ann, X, Y))   # ciclo 0

    for epoch in 1:maxEpochs
        # calcula gradientes directamente sobre ann
        grads = Flux.gradient(ann -> loss(ann, X, Y), ann)

        # actualiza la red con el optimizador
        Flux.update!(opt_state, ann, grads)

        # calcular pérdida actual
        current_loss = loss(ann, X, Y)
        push!(trainingLosses, Float32(current_loss))

        # --- criterios de parada ---
        if current_loss <= minLoss
            break
        end

        if length(trainingLosses) >= lossChangeWindowSize
            lossWindow = trainingLosses[end-lossChangeWindowSize+1:end]
            minLossValue, maxLossValue = extrema(lossWindow)
            if (maxLossValue - minLossValue) / minLossValue <= minLossChange
                break
            end
        end
    end

    return trainingLosses
end


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)

    Xraw, Yraw = trainingDataset

    X = Float32.(Xraw')        # trasponer X → columnas = instancias
    Y = Bool.(Yraw')           # trasponer Y → columnas = etiquetas

    ann = newClassCascadeNetwork(size(X,1), size(Y,1))   # red inicial sin ocultas
    trainingLosses = trainClassANN!(ann, (X, Y), false;  # primer entrenamiento
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

    for _ in 1:maxNumNeurons
        ann = addClassCascadeNeuron(ann; transferFunction=transferFunction)

        # entrenar solo últimas capas
        if indexOutputLayer(ann) > 2
            newLosses = trainClassANN!(ann, (X, Y), true;
                maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
                minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
            append!(trainingLosses, newLosses[2:end])   # evitar repetir el primer valor
        end

        # entrenar toda la red
        newLosses = trainClassANN!(ann, (X, Y), false;
            maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
            minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
        append!(trainingLosses, newLosses[2:end])
    end

    return (ann, trainingLosses)
end;

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    X, y = trainingDataset
    Y = reshape(y, :, 1)  # vector y → matriz columna
    return trainClassCascadeANN(maxNumNeurons, (X, Y);
        transferFunction=transferFunction,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    # trainingSet: N x m (N instancias por filas, m atributos por columnas)
    N = size(trainingSet, 1)
    # Calcula W = (1/N) * S^T * S
    W = (trainingSet' * trainingSet) ./ N
    # Convertir a Float32
    Wf = Array{Float32,2}(W)
    # Poner diagonal a 0 (se permite bucle en esta versión)
    for i in 1:size(Wf,1)
        Wf[i,i] = 0.0f0
    end
    return Wf
end

function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    # Convertir booleanos (0/1) a -1/1 sin bucles y llamar al método Real,2
    # (2 .* trainingSet) .- 1  -> valores en { -1, 1 } (tipo Bool promocionado a Int)
    realset = Float32.((2 .* trainingSet) .- 1)
    return trainHopfield(realset)
end

function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    # trainingSetNCHW: N x C x H x W (NCHW)
    N = size(trainingSetNCHW, 1)
    m = Int(prod(size(trainingSetNCHW)[2:end]))  # C*H*W
    # Convertir a matriz 2D: N filas (una por imagen), m columnas (atributos)
    training2d = reshape(trainingSetNCHW, N, m)
    # Convertir a -1/1 y llamar al método 2D
    realset2d = Float32.((2 .* training2d) .- 1)
    return trainHopfield(realset2d)
end


function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    # Convertir entrada a Float32
    Sf = Float32.(S)
    # Multiplicación por la matriz de pesos
    raw = ann * Sf
    # Umbralización con sign, garantizando solo {-1, 1}
    out = sign.(raw)
    return Float32.(out)
end

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    # Convertir de Bool a {-1, 1}
    realS = (2 .* Int8.(S)) .- 1
    # Llamar al método de Reales
    out_real = stepHopfield(ann, realS)
    # Convertir de {-1, 1} a Bool (>= 0 → true, < 0 → false)
    return out_real .>= 0
end



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
    noiseSet = copy(datasetNCHW)  

    indices = randperm(length(noiseSet))[1:Int(round(length(noiseSet) * ratioNoise))]

    # Invertir los bits seleccionados
    noiseSet[indices] .= .!noiseSet[indices]

    return noiseSet
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    cropSet = copy(datasetNCHW)  
    
    indices = (size(datasetNCHW, 4) - Int(floor(ratioCrop * size(datasetNCHW, 4))) + 1) : size(datasetNCHW, 4)

    # Poner esos bits a 0 (false)
    cropSet[:,:,:,indices] .= false

    return cropSet

end;

function randomImages(numImages::Int, resolution::Int)
    # Genera números aleatorios uniformes en [0,1)
    # y los convierte en Bool con .>= 0.5
    return rand(numImages, 1, resolution, resolution) .>= 0.5
end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    labels = unique(labelArray)                   # dígitos únicos
    N = length(labels)                            # número de dígitos
    C, H, W = size(imageArray)[2:4]              # canales, alto, ancho

    # Crear matriz de salida en formato NCHW
    outputArray = similar(imageArray, eltype(imageArray), N, C, H, W)

    for i in 1:N
        digit = labels[i]
        # seleccionar imágenes de ese dígito y promediar
        averaged = dropdims(mean(imageArray[labelArray .== digit, :, :, :], dims=1), dims=1)
        outputArray[i, :, :, :] .= averaged      # asignar a la matriz de salida
    end

    return (outputArray, labels)
end;

function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    N = size(imageArray, 1)                    # número de imágenes a clasificar
    outputs = fill(-1, N)                      # vector de salida inicializado a -1

    # --- bucle sobre las imágenes de la plantilla ---
    for i in 1:length(templateLabels)
        template = templateInputs[[i], :, :, :]                   # plantilla en formato NCHW
        indicesCoincidence = vec(all(imageArray .== template, dims=[3,4]))  # coincidencias por pixel
        outputs[indicesCoincidence] .= Int(templateLabels[i])          # asignar etiqueta
    end

    return outputs #forzar a q salgan enteros
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    trainX, trainY, testX, testY = loadMNISTDataset(datasetFolder; labels=labels, datasetType=Float32)
    templateX, templateY = averageMNISTImages(trainX, trainY)

    #Umbralizar train, test y plantillas
    trainB     = trainX     .>= threshold
    testB      = testX      .>= threshold
    templateB  = templateX  .>= threshold

    hop = trainHopfield(templateB)

    #Ejecutar Hopfield sobre train/test (en formato ±1 para evitar problemas de tipos)
    to_pm1(A) = Float32.(2 .* A .- 1)                 
    trainStates = runHopfield(hop, to_pm1(trainB))   
    testStates  = runHopfield(hop, to_pm1(testB))

    # Comparar píxel a píxel con las plantillas
    trainStatesB = trainStates .>= 0
    testStatesB  = testStates  .>= 0

    trainPred = classifyMNISTImages(trainStatesB, templateB, templateY)
    testPred  = classifyMNISTImages(testStatesB,  templateB, templateY)

    #Calcular exactitud
    trainAcc = sum(trainPred .== trainY) / length(trainY)
    testAcc  = sum(testPred  .== testY)  / length(testY)

    return (trainAcc, testAcc)
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
using Random: shuffle
predict(model, inputs::AbstractArray) = (outputs = MLJ.predict(model, MLJ.table(inputs)); return levels(outputs)[int(outputs)]; )



using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}



function batchInputs(batch::Batch)
    return batch[1]
end;

function batchTargets(batch::Batch)
    return batch[2]
end;

function batchLength(batch::Batch)
    return size(batch[1], 1)
end;

function selectInstances(batch::Batch, indices::Any)
    return (batchInputs(batch)[indices, :], batchTargets(batch)[indices])
end;

function joinBatches(batch1::Batch, batch2::Batch)
    return (vcat(batchInputs(batch1), batchInputs(batch2)), vcat(batchTargets(batch1), batchTargets(batch2)))
end;

function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    N = batchLength(dataset)
    idx = 1:N
    idx = shuffleRows ? shuffle(idx) : idx
    parts = Base.Iterators.partition(idx, batchSize)
    [selectInstances(dataset, collect(p)) for p in parts]
end;


function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )

    # --- Concatenar supportVectors (si tiene instancias) con el dataset (FIFO: supportVectors primero) ---
    N_support = batchLength(supportVectors)
    fullDataset = N_support > 0 ? joinBatches(supportVectors, dataset) : dataset

    # --- Crear modelo con la sintaxis pedida (usando LIBSVM.Kernel.*) ---
    kernel_symbol = kernel=="linear"  ? LIBSVM.Kernel.Linear  :
                    kernel=="rbf"     ? LIBSVM.Kernel.RadialBasis :
                    kernel=="poly"    ? LIBSVM.Kernel.Polynomial :
                    kernel=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing

    if kernel_symbol === nothing
        error("Kernel '$kernel' no reconocido. Usa: linear, rbf, poly, sigmoid.")
    end

    model = SVMClassifier(
        kernel = kernel_symbol,
        cost   = Float64(C),
        gamma  = Float64(gamma),
        degree = Int32(degree),
        coef0  = Float64(coef0)
    )

    # --- Crear machine sobre el conjunto concatenado (no sobre el dataset original sin soporte) ---
    Xtable = MLJ.table(batchInputs(fullDataset))
    ycat   = categorical(batchTargets(fullDataset))
    mach = machine(model, Xtable, ycat)

    # --- Entrenar ---
    MLJ.fit!(mach)

    # --- Extraer índices de vectores de soporte (ordenados) ---
    indicesNewSupportVectors = sort(mach.fitresult[1].SVs.indices)

    # --- Separar índices que pertenecen al supportVectors pasado y los del dataset original ---
    # Recuerda: en fullDataset, las primeras N_support filas corresponden a supportVectors pasado
    is_from_old_support = indicesNewSupportVectors .<= N_support
    old_support_indices = collect(indicesNewSupportVectors[is_from_old_support])               # indices en supportVectors pasado
    new_dataset_indices = collect(indicesNewSupportVectors[.!is_from_old_support] .- N_support) # indices en dataset original (ajustadas)

    # Asegurar tipo Integer
    old_support_indices = Int.(old_support_indices)
    new_dataset_indices = Int.(new_dataset_indices)

    # --- Construir el Batch de vectores de soporte resultantes ---
    # seleccionar instancias de supportVectors (puede ser vacío) y del dataset original (puede ser vacío)
    sv_from_old = isempty(old_support_indices) ? (Array{eltype(supportVectors[1]),2}(undef,0,size(supportVectors[1],2)), Array{eltype(supportVectors[2]),1}(undef,0)) : selectInstances(supportVectors, old_support_indices)
    sv_from_dataset = isempty(new_dataset_indices) ? (Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)), Array{eltype(dataset[2]),1}(undef,0)) : selectInstances(dataset, new_dataset_indices)

    # concatenar (primero los antiguos supportVectors, luego los del dataset original)
    supportBatch = joinBatches(sv_from_old, sv_from_dataset)

    return (mach, supportBatch, (old_support_indices, new_dataset_indices))
end


function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)

    if length(batches) == 0
        error("El vector de batches está vacío.")
    end

    # Inicializar supportVectors vacío con tipos adecuados (tomando tipos del primer batch)
    first_batch = batches[1]
    empty_support = ( Array{eltype(first_batch[1]),2}(undef,0,size(first_batch[1],2)),
                      Array{eltype(first_batch[2]),1}(undef,0) )

    currentSupport = empty_support
    last_mach = nothing

    # Único bucle permitido: iterar por los batches en orden y actualizar el support set
    for b in batches
        mach, newSupport, _ = trainSVM(b, kernel, C; degree=degree, gamma=gamma, coef0=coef0, supportVectors=currentSupport)
        currentSupport = newSupport
        last_mach = mach
    end

    return last_mach
end




 


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)

    X, y = loadStreamLearningDataset(datasetFolder)
    X = Float32.(X)
    y = Bool.(y)

    memory   = (X[1:windowSize, :], y[1:windowSize])
    rest     = (X[windowSize+1:end, :], y[windowSize+1:end])
    batchList = divideBatches(rest, batchSize; shuffleRows=false)
    return (memory, batchList)
end



function addBatch!(memory::Batch, newBatch::Batch)
    Xm, ym = memory          # Xm: N×M, ym: N
    Xn, yn = newBatch        # Xn: k×M, yn: k

    n = size(Xm, 1)
    k = size(Xn, 1)

    # Desplazar hacia el principio descartando los k más antiguos
    if n > k
        Xm[1:n-k, :] .= Xm[(k+1):n, :]
        ym[1:n-k]    .= ym[(k+1):n]
    end

    # Copiar al final el nuevo lote, respetando el orden temporal
    Xm[(n-k+1):n, :] .= Xn
    ym[(n-k+1):n]    .= yn

    return nothing
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
 
    (memory, batches) = initializeStreamLearningData(datasetFolder, windowSize, batchSize)
    if length(batches) == 0
        return Float64[]
    end

    model = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    model = model[1] #extraemos el machine
    accuracies = zeros(Float64, length(batches))

    for (i, batch) in enumerate(batches)
        preds = predict(model, batchInputs(batch))
        y_true = batchTargets(batch)
        accuracies[i] = sum(string.(preds) .== string.(y_true)) / length(y_true)
        addBatch!(memory, batch)
        if i < length(batches)
            model = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
            model = model[1] #extraemos el machine
        end
    end

    @assert isa(accuracies, Vector{<:Real})
    @assert length(accuracies) == length(batches)
    return accuracies
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    
    # Inicialización (warm start)
    initialBatch, batches = initializeStreamLearningData(datasetFolder, batchSize, batchSize)

    # Entrenamiento inicial (warm start)
    model, supportVectors, (_, indicesSupportVectorsInFirstBatch) =
        trainSVM(initialBatch, kernel, C; degree=degree, gamma=gamma, coef0=coef0)

    # Vector de edades para el initialBatch (descendente: batchSize ... 1)
    ages = collect(batchSize:-1:1)
    supportAges = isempty(indicesSupportVectorsInFirstBatch) ? Int[] : ages[indicesSupportVectorsInFirstBatch]

    # Vector de precisiones
    accuracies = zeros(Float64, length(batches))

    # Bucle principal (único bucle)
    for (i, batch) in enumerate(batches)
        # Evaluación
        preds = predict(model, batchInputs(batch))
        accuracies[i] = mean(preds .== batchTargets(batch))

        # Actualizar edades de los support vectors actuales (envejecer)
        supportAges .+= batchLength(batch)

        # Filtrar por antigüedad (mantener <= windowSize)
        valid_indices = findall(a -> a <= windowSize, supportAges)
        supportVectors = isempty(valid_indices) ? (Array{eltype(supportVectors[1]),2}(undef,0,size(supportVectors[1],2)), Array{eltype(supportVectors[2]),1}(undef,0)) : selectInstances(supportVectors, valid_indices)
        supportAges = supportAges[valid_indices]

        # Entrenar ISVM incremental con los supportVectors actuales
        model, _newSupportBatch, (idxOld, idxBatch) =
            trainSVM(batch, kernel, C; degree=degree, gamma=gamma, coef0=coef0, supportVectors=supportVectors)

        # Reconstruir nuevo conjunto de support vectors siguiendo el enunciado
        sv_from_old = isempty(idxOld)   ? (Array{eltype(supportVectors[1]),2}(undef,0,size(supportVectors[1],2)), Array{eltype(supportVectors[2]),1}(undef,0)) : selectInstances(supportVectors, Int.(idxOld))
        sv_from_batch = isempty(idxBatch) ? (Array{eltype(batch[1]),2}(undef,0,size(batch[1],2)), Array{eltype(batch[2]),1}(undef,0)) : selectInstances(batch, Int.(idxBatch))
        supportVectors = joinBatches(sv_from_old, sv_from_batch)

        # Actualizar vector de edades para los nuevos support vectors
        new_batch_ages = collect(batchLength(batch):-1:1)
        new_batch_ages_selected = isempty(idxBatch) ? Int[] : new_batch_ages[Int.(idxBatch)]
        supportAges = vcat( isempty(idxOld) ? Int[] : supportAges[Int.(idxOld)], new_batch_ages_selected )

        # Comprobación defensiva
        @assert length(supportAges) == batchLength(supportVectors)
    end

    return accuracies
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    # Diferencias: restar instance (traspuesta) a cada fila del dataset
    diffs = batchInputs(dataset) .- instance'          # size: N x M

    # Elevar al cuadrado y sumar por fila (dims=2)
    sumsq = sum(diffs.^2, dims=2)                     # size: N x 1

    # Raíz cuadrada y convertir a vector
    distances = vec(sqrt.(sumsq))                     # size: N

    return distances
end


function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    # calcular distancias (se asume que euclideanDistances devuelve un Vector{<:Real})
    dists = euclideanDistances(dataset, instance)

    n = length(dists)
    # si no hay instancias, devolver un Batch vacío mediante selectInstances con índices vacíos
    if n == 0
        return selectInstances(dataset, Int[])
    end

    # asegurar k en rango [1, n]
    k_clamped = clamp(k, 1, n)

    # obtener índices de los k más pequeños (no necesariamente ordenados)
    idx_k = partialsortperm(dists, 1:k_clamped)

    # ordenar esos índices por distancia ascendente (de más cercano a más lejano)
    idx_sorted = sort(idx_k, by = i -> dists[i])

    # seleccionar y devolver el sub-batch
    return selectInstances(dataset, idx_sorted)
end


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

