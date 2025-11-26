#!/usr/bin/env julia
# scripts/impute_subjectwise.jl
#
# Uso:
# julia --project=environment scripts/impute_sujeto.jl dataset.csv
#
# La salida se genera automáticamente como <input_basename>_imputed.csv
# Imputa valores missing por sujeto, usando media o mediana según outliers

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(joinpath(project_root, "environment"))

using CSV, DataFrames, Statistics, Dates

# ======================= ARGS ==========================
if length(ARGS) < 1
    println("Uso: julia scripts/impute_sujeto.jl <INPUT_CSV>")
    exit(1)
end

INPUT_CSV = ARGS[1]

if !isfile(INPUT_CSV)
    error("INPUT_CSV no existe: $INPUT_CSV")
end

# Salida automática
parent = dirname(INPUT_CSV)
name = splitext(basename(INPUT_CSV))[1]
OUTPUT_CSV = joinpath(parent, string(name, "_imputed.csv"))

println("Leyendo CSV: $INPUT_CSV")
df = DataFrame(CSV.File(INPUT_CSV; normalizenames=true))
println("Filas: ", nrow(df), " Columnas: ", ncol(df))
println("Inicio imputación subject-wise: ", Dates.now())

# ======================= Columnas numéricas ==========================
excluded = Set(["subject", "activity"])

function col_contains_numeric(eltyp)
    if eltyp <: Real
        return true
    end
    try
        for t in Base.uniontypes(eltyp)
            if t <: Real
                return true
            end
        end
    catch
    end
    return false
end

numeric_cols = String[]
for c in names(df)
    if c ∉ excluded && col_contains_numeric(eltype(df[!,c]))
        push!(numeric_cols, c)
    end
end


subjects = unique(df.subject)

# ======================= Outlier detection (IQR) =====================
function tiene_outliers(vals)
    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    any(x -> x < lower || x > upper, vals)
end

# ======================= IMPUTACIÓN ================================

for s in subjects
    df_s = df[df.subject .== s, :]
    total_missing = 0
    cols_imputed = String[]
    methods_used = String[]

    for col in numeric_cols
        colvec = df_s[!, col]
        nmiss = count(ismissing, colvec)
        if nmiss == 0
            continue
        end

        nonmiss = collect(skipmissing(colvec))
        if isempty(nonmiss)
            continue
        end

        method = tiene_outliers(nonmiss) ? "median" : "mean"
        value = method == "median" ? median(nonmiss) : mean(nonmiss)

        mask = (df.subject .== s) .& ismissing.(df[!, col])
        df[mask, col] .= value

    end

    
end

# ======================= GUARDAR CSV ==============================
println("\nGuardando CSV en: $OUTPUT_CSV")
CSV.write(OUTPUT_CSV, df)
println("Hecho. Fin: ", Dates.now())
