#!/usr/bin/env julia
# ML1/scripts/cv_individual_wise.jl
#
# Uso:
# julia --project=environment scripts/cv_individual_wise.jl /ruta/a/dataset_consolidado_train_imputed.csv
#
# Crea una validación cruzada individual-wise 5-Fold:
# - Cada fold tiene 5 sujetos distintos
# - 1 sujeto para validación, 4 para entrenamiento
# - Guarda CSVs por fold: *_train.csv y *_val.csv

using CSV, DataFrames, Random, Dates

if length(ARGS) < 1
    println("Uso: julia scripts/cv_individual_wise.jl <RUTA_CSV_TRAIN_IMPUTED>")
    exit(1)
end

DATA_PATH = ARGS[1]

println("Leyendo dataset train imputado...")
df = CSV.read(DATA_PATH, DataFrame)

# ------------------ sujetos disponibles ------------------
subjects = unique(df.subject)
println("Sujetos disponibles para CV: ", length(subjects))

# Semilla para reproducibilidad
Random.seed!(104)
shuffle!(subjects)

# Parámetros
fold_size = 5      # sujetos por fold
n_folds = 5
n_subjects = length(subjects)
n_used_subjects = fold_size * n_folds
unused_subjects = subjects[n_used_subjects+1:end]  # los que quedan fuera

println("Sujetos no asignados a folds: ", unused_subjects)

# Ruta base para guardar CSVs
base_path = replace(DATA_PATH, ".csv" => "")

# ------------------ creación de folds ------------------
println("\n=== Definición de folds (por sujeto) ===")
for fold_idx in 1:n_folds
    fold_subjects = subjects[(fold_idx-1)*fold_size + 1 : fold_idx*fold_size]
    println("Fold $fold_idx → Sujetos: ", fold_subjects)
    


    # Elegimos 1 sujeto para validación (primer sujeto) y 4 para entrenamiento
    val_subject = fold_subjects[1]
    train_subjects = fold_subjects[2:end]
    println("  -> Sujeto de validación (test) para este fold: ", val_subject)

    train_mask = in.(df.subject, Ref(train_subjects))
    val_mask   = in.(df.subject, Ref([val_subject]))

    df_train = df[train_mask, :]
    df_val   = df[val_mask, :]

    println("\nFold $fold_idx:")
    println(" - Train filas: ", nrow(df_train))
    println(" - Val   filas: ", nrow(df_val))
    println(" - Train sujetos: ", length(train_subjects))
    println(" - Val sujetos  : ", 1)

    # Guardar CSVs
    train_path = string(base_path, "_fold", fold_idx, "_train.csv")
    val_path   = string(base_path, "_fold", fold_idx, "_val.csv")
    CSV.write(train_path, df_train)
    CSV.write(val_path, df_val)
end

println("\nValidación cruzada individual-wise 5-Fold generada correctamente.")
println("Hecho: ", Dates.now())
