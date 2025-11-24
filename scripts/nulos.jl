#!/usr/bin/env julia
# MAAI/scripts/nulos.jl
#
# Uso:
# julia --project=environment scripts/nulos.jl /ruta/dataset_consolidado.csv
#
# Devuelve en pantalla:
# - Porcentaje de valores nulos por variable
# - Porcentaje total de valores nulos en todo el dataset
using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(joinpath(project_root, "environment"))

using CSV, DataFrames, Statistics

# ------------------ args ------------------
if length(ARGS) < 1
    println("Uso: julia scripts/nulos.jl <RUTA_CSV>")
    exit(1)
end

CSV_PATH = ARGS[1]

if !isfile(CSV_PATH)
    error("No se encontró el archivo CSV: $CSV_PATH")
end

# ------------------ lectura ------------------
println("Leyendo CSV consolidado...")
df = DataFrame(CSV.File(CSV_PATH; normalizenames=true))
println("Lectura completada: ", nrow(df), " filas, ", ncol(df), " columnas.")

# ------------------ análisis de nulos ------------------
n_rows = nrow(df)
n_cols = ncol(df)

# Porcentaje de nulos por columna
porc_nulos_col = Dict{String, Float64}()
for col in names(df)
    nulos = count(ismissing, df[!, col])
    porc_nulos_col[col] = 100 * nulos / n_rows
end

# Porcentaje total de nulos en todo el dataset
total_nulos = sum(count(ismissing, df[!, col]) for col in names(df))
total_valores = n_rows * n_cols
porc_total_nulos = 100 * total_nulos / total_valores

# ------------------ mostrar ------------------
println("\n=== Porcentaje de valores nulos por columna ===")
for (col, pct) in sort(collect(porc_nulos_col); by=x->x[2], rev=true)
    println(rpad(col, 30), ": ", round(pct, digits=2), "%")
end

println("\nPorcentaje total de valores nulos en el dataset: ", round(porc_total_nulos, digits=2), "%")
println("================================================")
