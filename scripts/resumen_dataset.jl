#!/usr/bin/env julia
# MAAI/scripts/resumen_dataset.jl
#
# Uso:
# julia --project=environment scripts/resumen_dataset.jl /ruta/dataset_consolidado.csv
#
# Devuelve en pantalla:
# - Número de variables
# - Número de instancias
# - Número de individuos
# - Número de clases de salida

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(joinpath(project_root, "environment"))

using CSV, DataFrames

# ------------------ args ------------------
if length(ARGS) < 1
    println("Uso: julia scripts/resumen_dataset.jl <RUTA_CSV>")
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
#println(names(df))


# ------------------ resumen ------------------
num_variables = ncol(df)
num_instancias = nrow(df)

# Suponiendo que la columna del sujeto se llama "subject" y la etiqueta de actividad "activity"
if "subject" ∈ names(df)
    num_individuos = length(unique(df[!,"subject"])) #devuelve una copia completa para lecutra y escritura
else
    @warn "No se encontró la columna 'subject', no se puede calcular número de individuos."
    num_individuos = missing
end

if "Activity" ∈ names(df)
    num_clases_salida = length(unique(df[!,"Activity"]))
else
    @warn "No se encontró la columna 'activity', no se puede calcular número de clases de salida."
    num_clases_salida = missing
end



# ------------------ mostrar ------------------
println("\n=== Resumen del dataset ===")
println("Número de variables: ", num_variables - 2) # Restamos 2 por "subject" y "activity" que es el target
println("Número de instancias: ", num_instancias)
println("Número de individuos: ", num_individuos)
println("Número de clases de salida: ", num_clases_salida)
println("===========================")
