#!/usr/bin/env julia
# ML1/scripts/cv_individual_wise.jl
#
# Uso:
# julia --project=environment scripts/cv_individual_wise.jl path/dataset_train.csv
#
# Genera 5 folds individual-wise (5-Fold CV) sin mezclar sujetos entre train y val.

using CSV, DataFrames, Random

if length(ARGS) < 1
    println("ERROR: Debes pasar el path al CSV train imputado.")
    exit(1)
end

train_path = ARGS[1]

println("Leyendo dataset train imputado...")
df = CSV.read(train_path, DataFrame)

# # subjects = lista de sujetos que quedan en el train (ya en tu CSV train)
subjects = unique(df.subject)
n = length(subjects)
println("Sujetos disponibles para CV: ", n)

# Semilla reproducible
Random.seed!(104)

# Barajar sujetos
shuffled_subjects = shuffle(subjects)

# Crear 5 folds
K = 5
folds = [shuffled_subjects[f:n:5] for f in 1:5]

println("\n=== Definición de folds (por sujeto) ===")
for (i, fold) in enumerate(folds)
    println("Fold $i → Sujetos: ", fold)
end

# Guardar particiones
outprefix = replace(train_path, ".csv" => "")

for i in 1:length(folds)
    val_subjects = folds[i]
    train_subjects = reduce(vcat, folds[setdiff(1:length(folds), [i])])

    # Máscaras
    val_mask = in.(df.subject, Ref(val_subjects))
    train_mask = in.(df.subject, Ref(train_subjects))

    df_train_fold = df[train_mask, :]
    df_val_fold = df[val_mask, :]

    fold_train_path = string(outprefix, "_cv_fold", i, "_train.csv")
    fold_val_path   = string(outprefix, "_cv_fold", i, "_val.csv")

    CSV.write(fold_train_path, df_train_fold)
    CSV.write(fold_val_path, df_val_fold)

    println("\nFold $i:")
    println(" - Train filas: ", nrow(df_train_fold))
    println(" - Val   filas: ", nrow(df_val_fold))
    println(" - Train usuarios: ", length(unique(df_train_fold.subject)))
    println(" - Val usuarios  : ", length(unique(df_val_fold.subject)))
end

println("\nValidación cruzada individual-wise 5-Fold generada correctamente.")
