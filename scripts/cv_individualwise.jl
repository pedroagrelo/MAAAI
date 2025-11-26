#!/usr/bin/env julia
# ML1/scripts/cv_individual_wise.jl
#
# Uso:
# julia --project=environment scripts/cv_individualwise.jl dataset_train_imputed.csv
#
# Crea un CV 5-fold basado en sujetos:
# - Cada fold usa un conjunto distinto de sujetos como test
# - El resto de sujetos se usan como train
# - No se mezclan sujetos entre train y test

using CSV, DataFrames, Random, Dates

if length(ARGS) < 1
    println("Uso: julia scripts/cv_individualwise.jl <RUTA_CSV_TRAIN_IMPUTED>")
    exit(1)
end

DATA_PATH = ARGS[1]
println("Leyendo dataset imputado…")
df = CSV.read(DATA_PATH, DataFrame)

# -----------------------------------
# Sujetos disponibles
# -----------------------------------
subjects = unique(df.subject)
n_subjects = length(subjects)
n_folds = 5

println("Sujetos encontrados: $n_subjects")


# -----------------------------------
# Función para generar folds balanceados
# -----------------------------------
function generate_folds(subjects::Vector{Int}, n_folds::Int=5)
    n_subjects = length(subjects)
    Random.seed!(104)
    shuffle!(subjects)

    base_size = div(n_subjects, n_folds)
    extra = n_subjects % n_folds

    folds = Vector{Vector{Int}}()
    start_idx = 1
    for i in 1:n_folds
        fold_size = base_size + (i <= extra ? 1 : 0)
        push!(folds, subjects[start_idx:start_idx+fold_size-1])
        start_idx += fold_size
    end
    return folds
end



folds = generate_folds(subjects, n_folds)

# -----------------------------------
# Guardar train/test por fold
# -----------------------------------

base_path = replace(DATA_PATH, ".csv" => "")

for i in 1:n_folds
    fold_subjects = folds[i]
    fold_subjects_shuffled = copy(fold_subjects)
    shuffle!(fold_subjects_shuffled)       # barajamos el fold
    test_subject = fold_subjects_shuffled[1]
    train_subjects = fold_subjects_shuffled[2:end]

    println("Fold   $i  :Sujetos en fold: ", fold_subjects)
    println("  -> Sujeto de validación (test): ", test_subject)
    println("  - Train sujetos: ", train_subjects)
    println("  - Train count: ", length(train_subjects))
    println("  - Test count : 1\n")

    test_mask = in.(df.subject, Ref([test_subject]))
    train_mask = in.(df.subject, Ref(train_subjects))

    df_train = df[train_mask, :]
    df_test  = df[test_mask, :]

    CSV.write("$(base_path)_fold$(i)_train.csv", df_train)
    CSV.write("$(base_path)_fold$(i)_test.csv", df_test)
end

println("\nValidación cruzada individual-wise 5-fold generada correctamente.")
println("Hora: ", Dates.now())
