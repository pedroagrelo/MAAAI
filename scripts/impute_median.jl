#!/usr/bin/env julia
# scripts/impute_median.jl
#
# Uso:
# julia --project=environment scripts/impute_median.jl ruta/dataset_consolidado.csv [ruta/salida_imputed.csv]
#
# Si no pasas OUTPUT, se creará <input_basename>_imputed.csv en el mismo directorio.

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(joinpath(project_root, "environment"))

using CSV, DataFrames, Statistics, Dates

# ---------- args ----------
if length(ARGS) < 1
    println("Uso: julia scripts/impute_median.jl <INPUT_CSV> [OUTPUT_CSV]")
    exit(1)
end

INPUT_CSV = ARGS[1]
OUTPUT_CSV = length(ARGS) >= 2 ? ARGS[2] : begin
    parent = dirname(INPUT_CSV)
    name = splitext(basename(INPUT_CSV))[1]
    joinpath(parent, string(name, "_imputed.csv"))
end

if !isfile(INPUT_CSV)
    error("INPUT_CSV no existe: $INPUT_CSV")
end

println("Leyendo CSV: ", INPUT_CSV)
df = DataFrame(CSV.File(INPUT_CSV; normalizenames=true))
println("Leído. Filas: ", nrow(df), " Columnas: ", ncol(df))
println("Inicio imputación con mediana: ", Dates.now())

# ---------- helper: detectar columnas numéricas ----------
function col_contains_numeric(eltyp)
    # si el eltype es Real o Union{Missing, Real} etc.
    if eltyp <: Real
        return true
    end
    # si es una unión (por ejemplo Union{Missing, Float64})
    try
        u = Base.uniontypes(eltyp)
        for t in u
            if t <: Real
                return true
            end
        end
    catch
        # Base.uniontypes falla si no es union; lo ignoramos
    end
    return false
end

# Columnas a proteger/excluir (no imputar)
excluded = Set(["subject", "activity"])

# Detectar columnas candidatas a imputar
numeric_cols = String[]
for name in names(df)
    if name in excluded
        continue
    end
    eltyp = eltype(df[!, name])
    if col_contains_numeric(eltyp)
        push!(numeric_cols, name)
    end
end

println("Columnas numéricas detectadas para imputar (excluyendo subject/activity):")
println(join(numeric_cols, ", "))

# ---------- imputación por mediana (corregido) ----------
impute_report = Vector{NamedTuple{(:col, :n_missing, :median)}}()

for col in numeric_cols
    colvec = df[!, col]
    nmiss = count(ismissing, colvec)
    if nmiss == 0
        push!(impute_report, (col=col, n_missing=0, median=missing))
        continue
    end

    # recoger valores no missing en un vector con nombre no ambiguo
    nonmiss_vals = collect(skipmissing(colvec))

    if isempty(nonmiss_vals)
        @warn "La columna $col tiene todos los valores missing; no se puede imputar."
        push!(impute_report, (col=col, n_missing=nmiss, median=missing))
        continue
    end

    # calcular mediana sobre los valores no missing
    med = median(nonmiss_vals)

    # reemplazar missing por la mediana
    df[!, col] = coalesce.(df[!, col], med)

    push!(impute_report, (col=col, n_missing=nmiss, median=med))
    println("Imputada columna: ", col, "  missing antes=", nmiss, "  mediana=", med)
end


# ---------- resumen ----------
println("----- Resumen de imputación -----")
for r in impute_report
    if r[:median] === missing
        println(r[:col], " -> missing antes=", r[:n_missing], "  mediana=ND (no imputado)")
    else
        println(r[:col], " -> missing antes=", r[:n_missing], "  mediana=", r[:median])
    end
end

# Guardar CSV resultante
println("Guardando CSV imputado en: ", OUTPUT_CSV)
CSV.write(OUTPUT_CSV, df)
println("Guardado. Fin imputación: ", Dates.now())
