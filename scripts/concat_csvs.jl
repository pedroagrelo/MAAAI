#!/usr/bin/env julia
# MAAI/scripts/concat_csvs.jl
#
# Uso:
# julia --project=environment scripts/concat_csvs.jl /ruta/a/datos /ruta/de/salida/dataset_consolidado.csv
#
# Si no se pasa OUTPUT_CSV, se crea "dataset_consolidado.csv" dentro de DATA_ROOT.

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(joinpath(project_root, "environment"))

using CSV, DataFrames, Dates, FilePathsBase

# ------------------ args ------------------
if length(ARGS) < 1
    println("Uso: julia scripts/concat_csvs.jl <DATA_ROOT> [OUTPUT_CSV]")
    exit(1)
end

DATA_ROOT = ARGS[1]
OUTPUT_CSV = length(ARGS) >= 2 ? ARGS[2] : joinpath(DATA_ROOT, "dataset_consolidado.csv")

if !isdir(DATA_ROOT)
    error("DATA_ROOT no existe o no es un directorio: $DATA_ROOT")
end

# ------------------ funciones ------------------
function gather_csvs(dir::AbstractString)
    csv_files = String[]

    function dfs_list(d)
        for entry in sort(readdir(d; join=true))
            if isdir(entry)
                dfs_list(entry)
            elseif endswith(lowercase(entry), ".csv")
                push!(csv_files, entry)
            end
        end
    end

    dfs_list(dir)
    return csv_files
end

# ------------------ ejecución ------------------
println("Inicio: ", Dates.now())
println("DATA_ROOT: ", DATA_ROOT)
println("OUTPUT CSV: ", OUTPUT_CSV)

# Si existe OUTPUT_CSV y quieres sobrescribir al ejecutar
rm(OUTPUT_CSV; force=true)

# Listar todos los CSVs en profundidad
files = gather_csvs(DATA_ROOT)
println("Se encontraron ", length(files), " archivos CSV.")

dfs_data = DataFrame[]

# Leer y almacenar todos los CSVs en memoria
for f in files
    try
        df = DataFrame(CSV.File(f; normalizenames=true))
        push!(dfs_data, df)
        println("Añadido ", basename(f), "  (", nrow(df), " filas)")
    catch e
        @warn "No se pudo leer CSV: $f — omitiendo." exception=(e, catch_backtrace())
    end
end

# Concatenar todos en un solo DataFrame
if !isempty(dfs_data)
    df_final = vcat(dfs_data...)
    CSV.write(OUTPUT_CSV, df_final)
    println("CSV consolidado generado en: ", OUTPUT_CSV)
else
    println("No se encontraron CSVs para concatenar.")
end

println("Finalizado: ", Dates.now())
