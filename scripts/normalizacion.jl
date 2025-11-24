#!/usr/bin/env julia
# ML1/scripts/normalizacion.jl
#
# Uso:
# julia --project=environment scripts/normalizar.jl /ruta/a/dataset.csv
#
# Aplica Min-Max scaling a columnas numéricas (excluye subject y Activity) y guarda CSV.

using CSV, DataFrames

if length(ARGS) < 1
    println("Uso: julia scripts/normalizar.jl <RUTA_CSV>")
    exit(1)
end

DATA_PATH = ARGS[1]

println("Leyendo dataset...")
df = CSV.read(DATA_PATH, DataFrame)

# Columnas a normalizar: numéricas excluyendo subject y Activity
numeric_cols = [c for c in names(df) if eltype(df[!, c]) <: Number && !(c in [:subject, :Activity])]

println("Columnas a normalizar: ", numeric_cols)

# Aplicar Min-Max scaling
for col in numeric_cols
    min_val = minimum(df[!, col])
    max_val = maximum(df[!, col])
    if max_val != min_val  # evitar división por cero
        df[!, col] = (df[!, col] .- min_val) ./ (max_val - min_val)
    else
        df[!, col] .= 0.0
    end
end

# Guardar CSV normalizado
out_path = replace(DATA_PATH, ".csv" => "_normalized.csv")
CSV.write(out_path, df)

println("Dataset normalizado guardado en: ", out_path)