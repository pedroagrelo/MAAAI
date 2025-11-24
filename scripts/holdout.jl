using CSV, DataFrames, Random, Dates

println("Leyendo dataset imputado...")
df = CSV.read("D:/Escritorio/Carpeta Universidade/2025-2026/MODELOS DE APRENDIZAJE AUTOMATICO/dataset_consolidado_imputed.csv", DataFrame)

println("Sujetos detectados:")
subjects = unique(df.subject)
println(subjects)

# Semilla pedida
Random.seed!(104)

# 10% de sujetos → round al menos a 1
n_test = max(1, round(Int, length(subjects) * 0.10))

println("Número total de sujetos: ", length(subjects))
println("Sujetos seleccionados para TEST (10%): ", n_test)

# Selección aleatoria reproducible
test_subjects = shuffle(subjects)[1:n_test]

println("\n=== Sujetos en el conjunto TEST ===")
println(test_subjects)

# Máscara de pertenencia
test_mask = in.(df.subject, Ref(test_subjects))
train_mask = .!test_mask

# Particionar
df_test  = df[test_mask, :]
df_train = df[train_mask, :]

println("\nFilas train: ", nrow(df_train))
println("Filas test : ", nrow(df_test))

# Prefijo para guardar
outpref = "D:/Escritorio/Carpeta Universidade/2025-2026/MODELOS DE APRENDIZAJE AUTOMATICO/dataset_consolidado_imputed"

# Guardar CSVs
train_path = string(outpref, "_train.csv")
test_path  = string(outpref, "_test.csv")
CSV.write(train_path, df_train)
CSV.write(test_path, df_test)

# Guardar lista de sujetos test
test_subjects_path = string(outpref, "_test_subjects.csv")
ts_df = DataFrame(subject = test_subjects)
CSV.write(test_subjects_path, ts_df)

# Informe por sujeto
counts = combine(groupby(df, :subject), nrow => :n_rows)
CSV.write(string(outpref, "_subject_counts.csv"), counts)

println("\nArchivos guardados:")
println(" - Train: ", train_path)
println(" - Test : ", test_path)
println(" - Test subjects list: ", test_subjects_path)
println(" - Subject counts: ", string(outpref, "_subject_counts.csv"))
println("Hecho: ", Dates.now())
