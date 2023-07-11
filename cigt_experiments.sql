--Experiments: Resnet CIGT Thin, No Branching.
--weight_decay = 1 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 18/2/2023
--Started on: HPC - /clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db

SELECT AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       A.Value AS ClassificationWd,
       COUNT(1) AS CNT
FROM logs_table
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Classification Wd") AS A ON
    logs_table.RunID = A.RunID
WHERE logs_table.RunID IN
(SELECT logs_table.RunId FROM logs_table LEFT JOIN run_meta_data ON
logs_table.RunId = run_meta_data.RunId
WHERE run_meta_data.Explanation LIKE "%Resnet CIGT Thin, No Branching.%") AND logs_table.Epoch >= 340
GROUP BY ClassificationWd


--Experiments: Resnet CIGT Vanilla - Always in warm up, no lr boosting.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 18/2/2023
--Started on: HPC - /clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db

SELECT logs_table.RunId,
       AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       AVG(A.Value) AS ClassificationWd,
       AVG(B.Value) AS InformationGainBalanceCoefficient,
       AVG(C.Value) AS DecisionLossCoefficient,
       AVG(D.Value) AS TemperatureDecayCoefficient,
       AVG(E.Value) AS InitialLr,
       COUNT(1) AS CNT
FROM logs_table
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Classification Wd") AS A ON
    logs_table.RunID = A.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Information Gain Balance Coefficient") AS B ON
    logs_table.RunID = B.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Decision Loss Coeff") AS C ON
    logs_table.RunID = C.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Temperature Decay Coefficient") AS D ON
    logs_table.RunID = D.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Initial Lr") AS E ON
    logs_table.RunID = E.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Use Straight Through") AS F ON
    logs_table.RunID = F.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Decision Nonlinearity") AS G ON
    logs_table.RunID = G.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "boostLearningRatesLayerWise") AS H ON
    logs_table.RunID = H.RunID WHERE logs_table.RunID IN
(SELECT logs_table.RunId FROM logs_table LEFT JOIN run_meta_data ON
logs_table.RunId = run_meta_data.RunId
WHERE run_meta_data.Explanation LIKE "%Resnet CIGT Vanilla - Always in warm up, no lr boosting.%")
AND logs_table.Epoch >= 340 AND F.Value = "True" AND G.Value = "Softmax"
GROUP BY logs_table.RunId
ORDER BY TestAccuracy DESC


--Experiments: Resnet CIGT Vanilla - Always in warm up, with lr boosting.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 18/2/2023
--Started on: HPC - /clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db

SELECT logs_table.RunId,
       AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       AVG(A.Value) AS ClassificationWd,
       AVG(B.Value) AS InformationGainBalanceCoefficient,
       AVG(C.Value) AS DecisionLossCoefficient,
       AVG(D.Value) AS TemperatureDecayCoefficient,
       AVG(E.Value) AS InitialLr,
       COUNT(1) AS CNT
FROM logs_table
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Classification Wd") AS A ON
    logs_table.RunID = A.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Information Gain Balance Coefficient") AS B ON
    logs_table.RunID = B.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Decision Loss Coeff") AS C ON
    logs_table.RunID = C.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Temperature Decay Coefficient") AS D ON
    logs_table.RunID = D.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Initial Lr") AS E ON
    logs_table.RunID = E.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Use Straight Through") AS F ON
    logs_table.RunID = F.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Decision Nonlinearity") AS G ON
    logs_table.RunID = G.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "boostLearningRatesLayerWise") AS H ON
    logs_table.RunID = H.RunID


WHERE logs_table.RunID IN
(SELECT logs_table.RunId FROM logs_table LEFT JOIN run_meta_data ON
logs_table.RunId = run_meta_data.RunId
WHERE run_meta_data.Explanation LIKE "%Resnet CIGT Vanilla - Always in warm up, with lr boosting.%")
AND logs_table.Epoch >= 420 AND F.Value = "True" AND G.Value = "Softmax"
GROUP BY logs_table.RunId
ORDER BY TestAccuracy DESC


--Experiments: Resnet CIGT Vanilla - Regular training.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 18/2/2023
--Started on: HPC - /clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db

SELECT logs_table.RunId,
       AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       AVG(A.Value) AS ClassificationWd,
       AVG(B.Value) AS InformationGainBalanceCoefficient,
       AVG(C.Value) AS DecisionLossCoefficient,
       AVG(D.Value) AS TemperatureDecayCoefficient,
       AVG(E.Value) AS InitialLr,
       COUNT(1) AS CNT
FROM logs_table
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Classification Wd") AS A ON
    logs_table.RunID = A.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Information Gain Balance Coefficient") AS B ON
    logs_table.RunID = B.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Decision Loss Coeff") AS C ON
    logs_table.RunID = C.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Temperature Decay Coefficient") AS D ON
    logs_table.RunID = D.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Initial Lr") AS E ON
    logs_table.RunID = E.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Use Straight Through") AS F ON
    logs_table.RunID = F.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Decision Nonlinearity") AS G ON
    logs_table.RunID = G.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "boostLearningRatesLayerWise") AS H ON
    logs_table.RunID = H.RunID


WHERE logs_table.RunID IN
(SELECT logs_table.RunId FROM logs_table LEFT JOIN run_meta_data ON
logs_table.RunId = run_meta_data.RunId
WHERE run_meta_data.Explanation LIKE "%Resnet CIGT Vanilla - Regular training.%")
AND logs_table.Epoch >= 340 AND F.Value = "True" AND G.Value = "Softmax"
GROUP BY logs_table.RunId
ORDER BY TestAccuracy DESC


--Experiments: Resnet Soft Routing.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 21/2/2023
--Started on: HPC - /clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db

--Experiments: Resnet Soft Routing - MoE Layer.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 21/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"
22|0.996252000001907|0.91997|0.0005|1.0|1.0|0.9999|0.1|10
21|0.995908000013351|0.91859|0.0005|1.0|1.0|0.9999|0.1|10
19|0.996366000003815|0.9182|0.0005|1.0|1.0|0.9999|0.1|10
20|0.995534000009537|0.91773|0.0005|1.0|1.0|0.9999|0.1|10
23|0.995022000013351|0.91436|0.0005|1.0|1.0|0.9999|0.1|10


--Experiments: Resnet Soft Routing - MoE Layer - Adam Optimizer.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 21/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger4.db"
5|0.997716000001907|0.90508|0.0005|1.0|1.0|0.9999|0.001|10
6|0.997526|0.90046|0.0005|1.0|1.0|0.9999|0.001|10
2|0.997652000003815|0.89857|0.0005|1.0|1.0|0.9999|0.001|10
3|0.997402000009537|0.89842|0.0005|1.0|1.0|0.9999|0.001|10
4|0.997642000001907|0.89751|0.0005|1.0|1.0|0.9999|0.001|10



--Experiments: Resnet Soft Routing - No MoE Layer - With Batch Statistics.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 21/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
30|0.997232000005722|0.92273|0.0005|1.0|1.0|0.9999|0.1|10
33|0.996482000001907|0.92001|0.0005|1.0|1.0|0.9999|0.1|10
32|0.994918000009537|0.91945|0.0005|1.0|1.0|0.9999|0.1|10
34|0.996094000005722|0.91823|0.0005|1.0|1.0|0.9999|0.1|10
31|0.995802000005722|0.918|0.0005|1.0|1.0|0.9999|0.1|10


--Experiments: Resnet Soft Routing - No MoE Layer - Entropy Regularization - Linear Routing Transformation.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 22/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
35|0.985506|0.90581|0.0005|1.0|1.0|0.9999|0.1|10
31|0.980502000015259|0.90054|0.0005|1.0|1.0|0.9999|0.1|10
33|0.895859999988556|0.83825|0.0005|1.0|1.0|0.9999|0.1|10
32|0.871608000007629|0.81522|0.0005|1.0|1.0|0.9999|0.1|10
34|0.728355999992371|0.7111|0.0005|1.0|1.0|0.9999|0.1|10

--Experiments: Resnet Soft Routing - No MoE Layer - Per Sample Entropy Regularization - Linear Routing Transformation.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 23/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
36|0.910194000005722|0.85854|0.0005|1.0|1.0|0.9999|0.1|10
38|0.756364|0.74808|0.0005|1.0|1.0|0.9999|0.1|10
37|0.462944000003815|0.40738|0.0005|1.0|1.0|0.9999|0.1|10
39|0.100000000002027|0.1|0.0005|1.0|1.0|0.9999|0.1|10
35|0.100000000002027|0.1|0.0005|1.0|1.0|0.9999|0.1|10


--Experiments: Resnet Soft Routing - Equal Probabilities.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 23/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"
29|0.999352000001907|0.93016|0.0005|1.0|1.0|0.9999|0.1|10
27|0.999248|0.92993|0.0005|1.0|1.0|0.9999|0.1|10
26|0.999096000001907|0.92844|0.0005|1.0|1.0|0.9999|0.1|10
25|0.998808|0.92655|0.0005|1.0|1.0|0.9999|0.1|10
28|0.999102000001907|0.9265|0.0005|1.0|1.0|0.9999|0.1|10



--Experiments: Resnet Soft Routing - Ideal Routing.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 23/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
43|0.998654000005722|0.95573|0.0005|1.0|1.0|0.9999|0.1|10
40|0.998928|0.95485|0.0005|1.0|1.0|0.9999|0.1|10
39|0.997422000003815|0.95338|0.0005|1.0|1.0|0.9999|0.1|10
41|0.998246000001907|0.95317|0.0005|1.0|1.0|0.9999|0.1|10
42|0.9984|0.95167|0.0005|1.0|1.0|0.9999|0.1|10


--Experiments: Resnet Soft Routing - Ideal Routing - Grid Search.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 23/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"

50|0.999738|0.94552|5.0e-05|1.0|1.0|0.9999|0.1|10
47|0.999574|0.9421|1.0e-05|1.0|1.0|0.9999|0.1|10
45|0.99968|0.94034|1.0e-05|1.0|1.0|0.9999|0.1|10
49|0.999642000001907|0.93899|1.0e-05|1.0|1.0|0.9999|0.1|10
41|0.999534000001907|0.93768|0.0|1.0|1.0|0.9999|0.1|10
43|0.999698000001907|0.93708|0.0|1.0|1.0|0.9999|0.1|10
44|0.99972|0.93444|0.0|1.0|1.0|0.9999|0.1|10
48|0.999654|0.93426|1.0e-05|1.0|1.0|0.9999|0.1|10
46|0.999596|0.93351|1.0e-05|1.0|1.0|0.9999|0.1|10
42|0.999710000001907|0.9322|0.0|1.0|1.0|0.9999|0.1|10
40|0.999486|0.93183|0.0|1.0|1.0|0.9999|0.1|10

--Experiments: Resnet Soft Routing - IG Routing.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 26/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"


--Experiments: Resnet Soft Routing - IG Routing - Different IG Losses.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 26/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"


--Experiments: Resnet Soft Routing - IG Routing - Iterative Training.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 26/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger4.db"

--Experiments: Resnet Soft Routing - IG Routing - Only IG Training.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 26/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"


--Experiments: Resnet Soft Routing - Variance Routing.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 28/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"


--Experiments: Resnet Hard Routing.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 28/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"

--Experiments: Resnet Hard Routing - With warm up.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 28/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"

--Experiments: Resnet Hard Routing - Only Routing.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 28/2/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger4.db"

--Experiments: Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,2.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 05/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
77|0.997300000001907|0.91708|0.0005|1.0|1.0|0.9999|0.1|10
75|0.996362000003815|0.91385|0.0005|1.0|1.0|0.9999|0.1|10
74|0.996274000005722|0.91348|0.0005|1.0|1.0|0.9999|0.1|10
78|0.994802000007629|0.91157|0.0005|1.0|1.0|0.9999|0.1|10
76|0.994802000001907|0.91117|0.0005|1.0|1.0|0.9999|0.1|10


--Experiments: Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,4.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 05/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
52|0.976810000009537|0.8951|0.0005|1.0|1.0|0.9999|0.1|10
53|0.883092000001907|0.82163|0.0005|1.0|1.0|0.9999|0.1|10
55|0.872594000005722|0.81846|0.0005|1.0|1.0|0.9999|0.1|10
56|0.775386000005722|0.72044|0.0005|1.0|1.0|0.9999|0.1|10
54|0.777436000011444|0.72028|0.0005|1.0|1.0|0.9999|0.1|10



--Experiments: Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,2. Batch Size 512.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 05/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"
53|0.999724000007629|0.92388999997139|0.0005|1.0|1.0|0.9999|0.1|10
51|0.999698000022888|0.921790000047684|0.0005|1.0|1.0|0.9999|0.1|10
52|0.999468000007629|0.920849999847412|0.0005|1.0|1.0|0.9999|0.1|10


--Experiments: Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,4. Batch Size 1024.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 07/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
57|0.999378400159073|0.912233999828338|0.0005|1.0|1.0|0.999899999999999|0.1|50


--Experiments: Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,3. Batch Size 1024.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 07/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"
55|0.898950000041199|0.829560000219345|0.0005|1.0|1.0|0.999899999999999|0.1|50


--Experiments: Resnet Hard Routing - 1,2,2. Random Routing Regularization. Batch Size 1024.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 08/3/2023
--Started on: HPC - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
9|0.999842291072923|0.922825931549072|0.0005|1.0|1.0|0.999899999999999|0.1|50
7|0.999838718686785|0.921291732311249|0.0005|1.0|1.0|0.999899999999999|0.1|50
10|0.999812877616104|0.917785993218422|0.0005|1.0|1.0|0.999899999999999|0.1|50
8|0.999857452742907|0.916083903908729|0.0005|1.0|1.0|0.999899999999999|0.1|50



--Experiments: Resnet Hard Routing - 1,2,4. Random Routing Regularization. Batch Size 1024.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 08/3/2023
--Started on: HPC - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
3|0.999726299363739|0.922110330820083|0.0005|1.0|1.0|0.999899999999999|0.1|50
2|0.999676294326782|0.918441963553429|0.0005|1.0|1.0|0.999899999999999|0.1|50
4|0.999661313027752|0.918347655653953|0.0005|1.0|1.0|0.999899999999999|0.1|50
1|0.999506890579146|0.918038463830948|0.0005|1.0|1.0|0.999899999999999|0.1|50

--Experiments: Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,2. Batch Size 1024.
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 07/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
82|0.999983600018311|0.924313999139786|0.0005|1.0|1.0|0.999899999999999|0.1|50
87|0.999972400009156|0.924067999357223|0.0005|1.0|1.0|0.999899999999999|0.1|50
85|0.999978|0.923919999404907|0.0005|1.0|1.0|0.999899999999999|0.1|50
88|0.999960400018311|0.923879999828339|0.0005|1.0|1.0|0.999899999999999|0.1|50
84|0.9999728|0.922363999300003|0.0005|1.0|1.0|0.999899999999999|0.1|50
86|0.9999696|0.919331999731064|0.0005|1.0|1.0|0.999899999999999|0.1|50
81|0.9999612|0.917663999597549|0.0005|1.0|1.0|0.999899999999999|0.1|50
0.922220285194124


--Experiments: Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,2. Batch Size 1024. - Classification Wd:0.00075
--weight_decay = 5 * [0.00075]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 07/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
61|0.999971600027466|0.927533999092102|0.00075|1.0|1.0|0.999899999999999|0.1|50 --0.93020000038147
63|0.999964800000001|0.926003999433518|0.00075|1.0|1.0|0.999899999999999|0.1|50
62|0.999978000009156|0.925731999361038|0.00075|1.0|1.0|0.999899999999999|0.1|50
60|0.999976000018311|0.924781999301911|0.00075|1.0|1.0|0.999899999999999|0.1|50
59|0.999972400027466|0.921429998975754|0.00075|1.0|1.0|0.999899999999999|0.1|50
0.925096399232864

--Experiments: Resnet Hard Routing - 1,2,2. Batch Size 1024. - Classification Wd:0.0008
--weight_decay = 5 * [0.0008]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 07/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
92|0.999975714318412|0.924999998937334|0.0008||1.0|0.9999|0.1|14
91|0.999965714318412|0.924842856195995|0.0008||1.0|0.9999|0.1|14
93|0.99997|0.923914284297398|0.0008||1.0|0.9999|0.1|14
90|0.999961428604126|0.923049999216625|0.0008||1.0|0.9999|0.1|14
89|0.999961428604126|0.922099999502727|0.0008||1.0|0.9999|0.1|14
0.923781427630016



--Experiments: Resnet Hard Routing - 1,2,2. Batch Size 1024. - Classification Wd:0.00085
--weight_decay = 5 * [0.00085]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 07/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
66|0.999964285746983|0.926514285911832|0.00085||1.0|0.9999|0.1|14
68|0.999971428571429|0.924807141903468|0.00085||1.0|0.9999|0.1|14
64|0.999961428571429|0.924571427631378|0.00085||1.0|0.9999|0.1|14
65|0.999965714285714|0.923135713522775|0.00085||1.0|0.9999|0.1|14
67|0.999961428604126|0.922535714912414|0.00085||1.0|0.9999|0.1|14
0.924312856776374

--Experiments: Resnet Hard Routing - 1,2,2. Batch Size 1024. - Classification Wd:0.0009
--weight_decay = 5 * [0.0009]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 07/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"

--Experiments: Resnet Hard Routing - 1,2,2. Batch Size 1024. - Classification Wd:0.00095
--weight_decay = 5 * [0.00095]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 07/3/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"
69|0.999961428571429|0.925678570617948|0.00095||1.0|0.9999|0.1|14
71|0.999984285714286|0.925042856822695|0.00095||1.0|0.9999|0.1|14
65|0.999961428604126|0.924478570188795|0.00095||1.0|0.9999|0.1|14
63|0.99996714288984|0.924199998950958|0.00095||1.0|0.9999|0.1|14
64|0.999955714318412|0.923849999209813|0.00095||1.0|0.9999|0.1|14
70|0.999948571428572|0.923792855923516|0.00095||1.0|0.9999|0.1|14
66|0.999962857175555|0.923164285060338|0.00095||1.0|0.9999|0.1|14
67|0.999940000032698|0.922414284985406|0.00095||1.0|0.9999|0.1|14
62|0.999942857142857|0.922099998664856|0.00095||1.0|0.9999|0.1|14
68|0.999938571428571|0.92205714242799|0.00095||1.0|0.9999|0.1|14
0.923677856285231


--Experiments: "Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,2. Batch Size 1024. Advanced Augmentation."
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 11/3/2023
--Started on: Tetam - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"
60|0.928768399950409|0.924985999296188|0.0005|1.0|1.0|0.999899999999999|0.1|50
57|0.932496799995422|0.920773999219894|0.0005|1.0|1.0|0.999899999999999|0.1|50
61|0.927537199983597|0.919981999305725|0.0005|1.0|1.0|0.999899999999999|0.1|50
59|0.929410400009155|0.91994999943924|0.0005|1.0|1.0|0.999899999999999|0.1|50
58|0.929331199992752|0.91827799969101|0.0005|1.0|1.0|0.999899999999999|0.1|50



--Experiments: "Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,4. Batch Size 1024. Balance Coefficients: [1.0, 5.0]. Standard Augmentation"
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 12/3/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
16|0.99941285733223|0.911521429204941|0.0005||1.0|0.9999|0.1|14
14|0.999614285845075|0.911035714742116|0.0005||1.0|0.9999|0.1|14
13|0.999555714416504|0.910621428101403|0.0005||1.0|0.9999|0.1|14
17|0.999550000163487|0.910371427842549|0.0005||1.0|0.9999|0.1|14
15|0.99956000009128|0.906749999727522|0.0005||1.0|0.9999|0.1|14


--Experiments: "Resnet Hard Routing. 1,2,4. Batch Size 1024. Balance Coefficients: [1.0, 5.0]. Standard Augmentation - Classification Wd:0.00075"
--weight_decay = 5 * [0.00075]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 12/3/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
18|0.999342857122421|0.908678571244649|0.00075||1.0|0.9999|0.1|14
19|0.999085714526858|0.906857142754964|0.00075||1.0|0.9999|0.1|14



--Experiments: "Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,4. Batch Size 1024. Balance Coefficients: [5.0, 5.0]. Standard Augmentation"
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 12/3/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
10|0.999482857430322|0.914399999461855|0.0005||1.0|0.9999|0.1|14
6|0.999441428721292|0.913292856836319|0.0005||1.0|0.9999|0.1|14
7|0.999472857273647|0.910528570829119|0.0005||1.0|0.9999|0.1|14
8|0.999560000196184|0.909457142754964|0.0005||1.0|0.9999|0.1|14
9|0.997880000186648|0.906278571523939|0.0005||1.0|0.9999|0.1|14

--Experiments: "Resnet Hard Routing. 1,2,4. Batch Size 1024. Balance Coefficients: [5.0, 5.0]. Standard Augmentation - Classification Wd:0.00075"
--weight_decay = 5 * [0.00075]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 12/3/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
12|0.99927571427209|0.911457142904826|0.00075||1.0|0.9999|0.1|14
11|0.999222857279096|0.910178570999418|0.00075||1.0|0.9999|0.1|14


--Experiments: "Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,4. Batch Size 1024. Balance Coefficients: [1.0, 5.0]. Advanced Augmentation"
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 12/3/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
5|0.929791428550993|0.923464284603936|0.0005||1.0|0.9999|0.1|14
3|0.918055714148113|0.920578571122033|0.0005||1.0|0.9999|0.1|14
2|0.921645714210783|0.920307143313544|0.0005||1.0|0.9999|0.1|14
4|0.918541428691319|0.919307142584664|0.0005||1.0|0.9999|0.1|14
1|0.924847142765863|0.918785713781629|0.0005||1.0|0.9999|0.1|14

--Experiments: "Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,4. Batch Size 1024. Balance Coefficients: [1.0, 5.0]. Advanced Augmentation. Classification Wd:0.0"
--weight_decay = 5 * [0.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 12/3/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
7|0.890921428552355|0.888235713134493|0.0||1.0|0.9999|0.1|14
6|0.89139571424893|0.886742856897627|0.0||1.0|0.9999|0.1|14



--Experiments: "Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,4. Batch Size 1024. Balance Coefficients: [5.0, 5.0]. Advanced Augmentation"
--weight_decay = 5 * [0.0005]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 12/3/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
2|0.922420000118528|0.922614285469055|0.0005||1.0|0.9999|0.1|14
4|0.919444285736084|0.922321428803035|0.0005||1.0|0.9999|0.1|14
0|0.916751428511483|0.921492856877191|0.0005||1.0|0.9999|0.1|14
3|0.927192857052939|0.921142855991636|0.0005||1.0|0.9999|0.1|14
1|0.926791428570066|0.918607142257691|0.0005||1.0|0.9999|0.1|14


--Experiments: "Resnet Hard Routing - Only Routing - Temperature Reset Fixed. 1,2,4. Batch Size 1024. Balance Coefficients: [5.0, 5.0]. Advanced Augmentation. Classification Wd:0.0"
--weight_decay = 5 * [0.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--Started at 12/3/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
5|0.894412857091086|0.890835715198517|0.0||1.0|0.9999|0.1|14
6|0.889635714385169|0.887392858178275|0.0||1.0|0.9999|0.1|14


--Experiments: "Cigt - [1,2,4] - Batch Size:1024 - Random Distribution Augmentation in Warm up: Classification Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--Started at 9/4/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
32|0.9226


--Experiments: "Cigt - [1,2,4] - Batch Size:1024 - MultipleLogitsMultipleLosses - information_gain_balance_coeff_list = [1.0, 1.0] - initial_lr = 0.025 - Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [1.0, 1.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--Started at 11/4/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
0.188200000056624|34
0.90710000050664|36
0.8983999994874|37
0.906800003516674|38
0.742900001186132|39



--Experiments: "Cigt - [1,2,4] - Batch Size:1024 - MultipleLogitsMultipleLosses - information_gain_balance_coeff_list = [5.0, 5.0] - initial_lr = 0.025 - Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--Started at 11/4/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
0.902499999409914|21
0.909099999970198|22
0.902899999564886|23
0.856200006043911|24



--Experiments: "Cigt - [1,2,4] - epoch_count:350 - Batch Size:1024 - MultipleLogitsMultipleLosses - information_gain_balance_coeff_list = [5.0, 5.0] - initial_lr = 0.1 - Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--Started at 11/4/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
0.897800000315905|15
0.901700002104044|16
0.879599997425079|17
0.893300001490116|18
0.883199995130301|19


--Experiments: "Cigt - [1,2,4] - epoch_count:350 - Batch Size:1024 - MultipleLogitsMultipleLosses - information_gain_balance_coeff_list = [5.0, 5.0] - initial_lr = 0.025 - Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--Started at 11/4/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
0.876699997383356|10
0.877099999326468|11
0.877900000780821|12
0.883200002115965|13
0.880100001263618|14


--Experiments: "Cigt - [1,2,4] - Batch Size:2048 - MultipleLogitsMultipleLosses - information_gain_balance_coeff_list = [5.0, 5.0] - initial_lr = 0.1 - Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--Started at 13/4/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
21|0.892899994394183
22|0.880600004374981
23|0.89180000448823
24|0.920599996888638



--Experiments: "Cigt - [1,2,4] - Batch Size:1024 - MultipleLogitsMultipleLosses - information_gain_balance_coeff_list = [5.0, 5.0] - initial_lr = 0.1 - Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--Started at 13/4/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
16|0.86940000115037
17|0.917300002145767
18|0.781399991622567
19|0.884800002676249
20|0.771100003343821


--Experiments: "Cigt - [1,2,4] - Batch Size:1024 - MultipleLogitsMultipleLossesAveraged - information_gain_balance_coeff_list = [5.0, 5.0] - initial_lr = 0.1 - Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLossesAveraged"
--Started at 16/4/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
41|0.900599998807907
42|0.884799997109175

--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--Started at 21/4/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005%";
--SELECT * FROM run_kv_store WHERE RunID = 45 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 45 AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 45 AND Key LIKE "%Path Distribution%";
--SELECT * FROM logs_table WHERE RunID = 45;
--SELECT RunID,Max(TestAccuracy) FROM logs_table WHERE RunID IN (45,46,47,48,49) GROUP BY RunID;
45|0.885300000858307
46|0.917299996739626
47|0.897299998694658
48|0.866599998682737
49|0.919199996173382



--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - With 350 Epoch WarmUp"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 21/4/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - With 350 Epoch WarmUp%";
--SELECT * FROM logs_table WHERE RunID = 29;
--SELECT * FROM run_kv_store WHERE RunID = 29 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 29 AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 29 AND Key LIKE "%Path Distribution%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (29,30,31,32,33) GROUP BY RunID;
29|0.9324
30|0.930400000286102
31|0.929399997806549
32|0.931100000095367
33|0.930200002098084




--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - With 350 Epoch WarmUp"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 21/4/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - With 350 Epoch WarmUp%";
--SELECT * FROM logs_table WHERE RunID = 95;
--SELECT * FROM run_kv_store WHERE RunID = 95 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 95 AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 95 AND Key LIKE "%Path Distribution%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (95,96,97,98,99) GROUP BY RunID;
95|0.934000000286102
96|0.929299997997284
97|0.9321
98|0.931300000095367
99|0.926899997901916




--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - Starting from random_cigtlogger2_29_epoch350.pth"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--warm_up_period = adjust_to_batch_size(original_value=0, target_batch_size=batch_size)
--Started at 23/4/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - Starting from random_cigtlogger2_29_epoch350.pth%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (29,30,31,32,33,34,35) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 31 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
30|0.933299997711182
31|0.923999998766184
32|0.869099996548891
33|0.828100002974272
34|0.81509999691844
35|0.882700001788139

-- THIS EXPERIMENT IS REPEATED FOR x3 TIMES!!!
--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 23/4/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (23,24,25,26,27,28,29,30,31,32,33,34,35,36,37) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 23 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT AVG(Value),MIN(Value),MAX(Value),COUNT(*) AS CNT FROM run_kv_store WHERE RunID = 23 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND 24549 <= Iteration AND Iteration <= 34594;
--SELECT Epoch,TestAccuracy FROM logs_table WHERE RunID = 23 ORDER BY TestAccuracy DESC;
23|0.925400000804663
24|0.902600006085634
25|0.922399997961521
26|0.906099998432398
27|0.923400000238419
28|0.924700003921986
29|0.906899998790026
30|0.925499999910593
31|0.921800000411272
32|0.927200002264976
33|0.907600001072884
34|0.927900001162291
35|0.923300000852346
36|0.906699999773502
37|0.925700000143051





--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.00075 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled"
--weight_decay = 5 * [0.00075]
--information_gain_balance_coeff_list = [5.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 30/4/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.00075 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (36,37,38,39,40) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 37 AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 37 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
36|0.805399998396635
37|0.797499998915196
38|0.885299997478724
39|0.922599999302626
40|0.924200008106232



--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - information_gain_balance_coeff_list:[1.0, 5.0]"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [1.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 30/4/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - information_gain_balance_coeff_list:[1.0, 5.0]%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (50,51,52,53,54) GROUP BY RunID;
50|0.904799998873472
51|0.92500000295043
52|0.923900002413988
53|0.926800004541874
54|0.925200001400709




--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.00075 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - information_gain_balance_coeff_list:[1.0, 5.0]"
--weight_decay = 5 * [0.00075]
--information_gain_balance_coeff_list = [1.0, 5.0]
--param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
--hard_routing_algorithm_kind = "InformationGainRouting"
--warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--loss_calculation_kind = "MultipleLogitsMultipleLosses"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 30/4/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.00075 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - information_gain_balance_coeff_list:[1.0, 5.0]%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (34,35,36,37,38) GROUP BY RunID;
34|0.860600000876188
35|0.887599998265505
36|0.922399999344349
37|0.916499997401237
38|0.925200002741814


--X2
--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 1/5/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (101) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 103 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT AVG(Value),MIN(Value),MAX(Value),COUNT(*) AS CNT FROM run_kv_store WHERE RunID = 102 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND 24549 <= Iteration AND Iteration <= 34594;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (100,101,102,103,104,105,106,107) GROUP BY RunID;
100|0.832600000092387
101|0.821799998956919
102|0.921600004523992
103|0.93099999833703
104|0.923000003921986
105|0.926800002121925
106|0.930199999338388
107|0.928300000739098


--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0006]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/5/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (108,109,110,111,112) GROUP BY RunID;
108|0.927399997335672
109|0.924799997907877
110|0.924000000017881
111|0.927399999552965
112|0.927699999564886



--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0007 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0007]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/5/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0007 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (69,70,71,72,73,74,75) GROUP BY RunID;
69|0.926199996364117
70|0.923500002747774
71|0.923599995863438
72|0.927199998307228
73|0.927199997669458
74|0.924200001811981
75|0.826799999672174



--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0008 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0008]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/5/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0008 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (72,73,74,75,76,77,78) GROUP BY RunID;
72|0.921700006115437
73|0.923900000470877
74|0.917600001168251
75|0.923099998795986
76|0.923399999517202
77|0.921700004220009
78|0.92439999704957


--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0001 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0001]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/9/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0001 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (113,114,115,116) GROUP BY RunID;
113|0.915000001376867
114|0.909400000798702
115|0.91459999781847
116|0.812799997395277



--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0002 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0002]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/9/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger3.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0002 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (79,80,81) GROUP BY RunID;
79|0.923300001782179
80|0.923099997574091
81|0.92170000077486



--Experiments: "Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/9/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (55,56,57,58,59) GROUP BY RunID;
55|0.92689999614954
56|0.925500004029274
57|0.929399994492531
58|0.924499992024899
59|0.930599988240004



--Experiments: "Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.00075 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.00075]
--information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/9/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.00075 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (39,40,41,42,43) GROUP BY RunID;
39|0.927199996709824
40|0.926199991643429
41|0.929999995148182
42|0.925700000357628
43|0.928499998348951


--Experiments: "Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0004 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0004]
--information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/9/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0004 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (41,42,43,44,45) GROUP BY RunID;
41|0.928399997979403
42|0.928700006991625
43|0.925799999839067
44|0.929599998939037
45|0.926800000292063



--Experiments: "Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0006]
--information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/9/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (38,39,40,41,42) GROUP BY RunID;
--SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (38) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000;
--SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (38) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000;
38|0.93239999216795
39|0.927300001370907
40|0.928899997699261
41|0.928700003737211
42|0.928299992477894



--Experiments: "Cigt Ideal Routing - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005"
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRouting"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=0, target_batch_size=batch_size)
--Started at 4/14/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt Ideal Routing - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (117,118,119,120,121) GROUP BY RunID;
117|0.961299999564886
118|0.961900001418591
119|0.960799999153614
120|0.961100001466274
121|0.961199999451637



--Experiments: "Cigt Ideal Routing - [1,2,4] - SingleLogitSingleLoss - Wd:0.0005"
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "SingleLogitSingleLoss"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRouting"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=0, target_batch_size=batch_size)
--Started at 4/14/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt Ideal Routing - [1,2,4] - SingleLogitSingleLoss - Wd:0.0005%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (76,77,78,79,80,81) GROUP BY RunID;
77|0.963899998950958
78|0.965400001525879
79|0.962200001621246
80|0.959700001335144
81|0.961800001621246


--Experiments: "Cigt - [1,2,4] - SingleLogitSingleLoss - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "SingleLogitSingleLoss"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/14/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - SingleLogitSingleLoss - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (60,61,62,63,64) GROUP BY RunID;
60|0.925799997997284
61|0.925399998188019
62|0.917900000476837
63|0.92560000038147
64|0.914500000858307


--Experiments: "Cigt - [1,2,2] - SingleLogitSingleLoss - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0006]
--information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "SingleLogitSingleLoss"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/9/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,2] - SingleLogitSingleLoss - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (44,45,46,47,48) GROUP BY RunID;



--Experiments: "KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.5"
--ResnetCigtConstants.use_kd_for_routing = False
--ResnetCigtConstants.kd_teacher_temperature = 6.0
--ResnetCigtConstants.kd_loss_alpha = 0.5
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/20/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.5%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (122,123,124) GROUP BY RunID;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (122,123,124,125,126,127,128) GROUP BY RunID;
124|0.922199996948242
125|0.927799996364117
126|0.929099997121096
127|0.926900000560284
128|0.929900004726648




--Experiments: "KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = True - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.5"
--ResnetCigtConstants.use_kd_for_routing = True
--ResnetCigtConstants.kd_teacher_temperature = 6.0
--ResnetCigtConstants.kd_loss_alpha = 0.5
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/20/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = True - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.5%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (82,83,84) GROUP BY RunID;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (82,83,84,85,86,87) GROUP BY RunID;
83|0.924099999654293
84|0.929200001996756
85|0.927600001722574
86|0.927800001972914
87|0.926399998110533




--Experiments: "KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.95"
--ResnetCigtConstants.use_kd_for_routing = False
--ResnetCigtConstants.kd_teacher_temperature = 6.0
--ResnetCigtConstants.kd_loss_alpha = 0.95
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/20/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.95%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (46, 47, 48, 49, 50) GROUP BY RunID;
46|0.924799997514486
47|0.922499998950958
48|0.923699995368719
49|0.923900002580881
50|0.926699997097254





--Experiments: "KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = True - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.95"
--ResnetCigtConstants.use_kd_for_routing = True
--ResnetCigtConstants.kd_teacher_temperature = 6.0
--ResnetCigtConstants.kd_loss_alpha = 0.95
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/20/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = True - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.95%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (43) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 43 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (43, 44, 45, 46, 47) GROUP BY RunID;
43|0.92780000462532
44|0.924600001174211
45|0.930199997997284
46|0.927099996614456
47|0.925000005412102




--Experiments: "KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.05"
--ResnetCigtConstants.use_kd_for_routing = False
--ResnetCigtConstants.kd_teacher_temperature = 6.0
--ResnetCigtConstants.kd_loss_alpha = 0.05
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/20/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 6.0 - kd_loss_alpha = 0.05%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (65, 66) GROUP BY RunID;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (65, 66, 67, 68, 69) GROUP BY RunID;
65|0.924999998432398
66|0.923300004184246
67|0.926199997234344
68|0.925600004351139
69|0.923099998015165



--Experiments: "KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 10.0 - kd_loss_alpha = 0.95"
--ResnetCigtConstants.use_kd_for_routing = False
--ResnetCigtConstants.kd_teacher_temperature = 10.0
--ResnetCigtConstants.kd_loss_alpha = 0.95
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/25/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 10.0 - kd_loss_alpha = 0.95%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (129, 130) GROUP BY RunID;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (129, 130, 131, 132, 133) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 129 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
129|0.922600003618002
130|0.925800003504753
131|0.924300000905991
132|0.923999993056059
133|0.925300002729893



--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_relu_dropout_to_decision_layer: True - decision_drop_probability: 0.5"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = True
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--Started at 4/30/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_relu_dropout_to_decision_layer: True - decision_drop_probability: 0.5%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (134, 135, 136, 137, 138) GROUP BY RunID;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (134, 135, 136, 137, 138) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 136 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT RunId, MIN(Value) FROM run_kv_store WHERE RunID IN (134, 135, 136, 137, 138) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" GROUP BY RunID;
134|0.924600000733137
135|0.924700002592802
136|0.927000001657009
137|0.928399997711182
138|0.926100002360344


SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (102,103,104,105,106,107) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000 GROUP BY RunID;
SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (134, 135, 136, 137, 138) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000 GROUP BY RunID;
134|-9.86517217697636
135|-9.86580379239974
136|-9.90531657126642
137|-8.80130875802809
138|-9.873926765688

SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (102,103,104,105,106,107) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000;
SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (134, 135, 136, 137) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000;
SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (102,103,104,105,106,107) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000;
SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (134, 135, 136, 137, 138) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000 GROUP BY RunID;

SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (102,103,104,105,106,107) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000 GROUP BY RunID;
SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (134, 135, 136, 137) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000 GROUP BY RunID;

SELECT RunId,MIN(Value) FROM run_kv_store WHERE RunID >= 102 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000;
134|0.924600000733137
135|0.924700002592802
136|0.927000001657009
137|0.928399997711182
138|0.926100002360344


--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_relu_dropout_to_decision_layer: True - decision_drop_probability: 0.5 - decision_dimensions = [64, 64]"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = True
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [64, 64]
--Started at 4/30/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_relu_dropout_to_decision_layer: True - decision_drop_probability: 0.5 - decision_dimensions = [64, 64]%";



--Experiments: "Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_relu_dropout_to_decision_layer: True - decision_drop_probability: 0.5 - decision_dimensions = [32, 32]"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = True
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [32, 32]
--Started at 4/30/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_relu_dropout_to_decision_layer: True - decision_drop_probability: 0.5 - decision_dimensions = [32, 32]%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (49) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 49 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (49) GROUP BY RunID;
--SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (49, 50, 51, 52, 53) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000 GROUP BY RunID;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (49, 50, 51, 52, 53) GROUP BY RunID;
49|0.924299998188019
50|0.928699997901916
51|0.929800000667572
52|0.92940000038147
53|0.832100001811981


--Experiments: "Gather Scatter Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = True
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [128, 128]
--Started at 6/03/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (139, 140, 141, 142, 143, 144) GROUP BY RunID;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (139, 140, 141, 142, 143, 144) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 140 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 140 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";

--SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (140, 141, 142, 143, 144) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000 GROUP BY RunID;
--SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (140, 141, 142, 143, 144) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000 GROUP BY RunID;
140|0.927700002133846
141|0.928000000089407
142|0.925499998474121
143|0.925600000739098
144|0.928200002563

sqlite> SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (140, 141, 142, 143, 144) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000 GROUP BY RunID;
140|-10.1521979734409
141|-10.1253156272733
142|-10.141199065479
143|-10.1419400245913
144|-10.1491592311796
sqlite> SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (140, 141, 142, 143, 144) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000 GROUP BY RunID;
140|-9.88043455000847
141|-9.872516422887
142|-9.85758160314253
143|-9.86383955555577
144|-9.87977511805873




--Experiments: "Gather Scatter Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = True
--weight_decay = 5 * [0.0006]
--information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [128, 128]
--Started at 6/03/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (89, 90, 91) GROUP BY RunID;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (89, 90, 91, 92, 93) GROUP BY RunID;
--SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (89, 90, 91) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000 GROUP BY RunID;
89|0.932100008404255
90|0.926800001519918
91|0.924499994122982
92|0.929900001275539
93|0.927399993145466







--Experiments: "Gather Scatter Cigt - [1,2,4] - [5.0, 5.0] - SingleLogitSingleLoss - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = True
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "SingleLogitSingleLoss"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [128, 128]
--Started at 6/03/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt - [1,2,4] - [5.0, 5.0] - SingleLogitSingleLoss - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (70, 71, 72, 73, 74) GROUP BY RunID;
70|0.925499998092651
71|0.924299998092651
72|0.92250000038147
73|0.924100000572205
74|0.920200000476837



--Experiments: "Gather Scatter Cigt - [1,2,2] - [1.0, 1.0] - SingleLogitSingleLoss - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = True
--weight_decay = 5 * [0.0006]
--information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "SingleLogitSingleLoss"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [128, 128]
--Started at 6/03/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt - [1,2,2] - [1.0, 1.0] - SingleLogitSingleLoss - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (54, 55, 56, 57, 58) GROUP BY RunID;
54|0.920699998378754
55|0.923099998188019
56|0.92440000038147
57|0.921700000762939
58|0.922000000858307



--Experiments: "Masked Cigt - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_mask_to_batch_norm: False"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [128, 128]
--Started at 6/08/2023
--Started on: Tetam - Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Masked Cigt - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_mask_to_batch_norm: False%";
--SELECT * FROM run_kv_store WHERE RunID = 75 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM logs_table WHERE RunID = 75;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (75, 76, 77, 78, 79) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (75, 76, 77, 78, 79) GROUP BY RunID;
75|0.893199997782707
76|0.901699999183416
77|0.906000000417232
78|0.900999998944998
79|0.9075999969244





--Experiments: "Masked Cigt - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_mask_to_batch_norm: True"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [128, 128]
--Started at 6/08/2023
--Started on: Tetam - Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Masked Cigt - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization - apply_mask_to_batch_norm: True%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (59) GROUP BY RunID;
--SELECT * FROM logs_table WHERE RunID = 59;
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (59, 60, 61, 62, 63) GROUP BY RunID;
59|0.913299999272823
60|0.904899994844198
61|0.904000002491474
62|0.905599996221066
63|0.905400001162291



--X3
--Experiments: "Gather Scatter Cigt With Random Augmentation - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [128, 128]
--Started at 6/08/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"

-- TUNA 1
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With Random Augmentation - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62) GROUP BY RunID;
--SELECT * FROM logs_table WHERE RunID = 51;
--SELECT * FROM run_kv_store WHERE RunID = 62 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 51 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";
--SELECT RunID, Max(Epoch), Max(Iteration) FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64) GROUP BY RunID;
--SELECT RunID, MIN(Value) FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID;
9.91252241134644

SELECT RunID, Max(TestAccuracy) AS Acc FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66) GROUP BY RunID ORDER BY Acc DESC;
65|0.938800000053644
56|0.937799996507168
60|0.937100003141165
54|0.93679999807477
59|0.936600001531839
52|0.935900000613928
57|0.935500001275539
66|0.935499999791384
55|0.935399998033047
58|0.935099999380112
62|0.934400002837181
53|0.934399996191263
64|0.934000003087521
51|0.933399998909235
63|0.932699999696016
61|0.923899998474121




SELECT RunID, MIN(Value) AS Score FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID ORDER BY Score ASC;
51|-9.93302793502808
56|-9.92644653320312
62|-9.92495069503784
58|-9.92368068695068
54|-9.92163553237915
52|-9.91686754226685
60|-9.91252241134644
64|-9.91120090484619
55|-9.91062927246094
65|-9.90777034759521
53|-9.90768098831177
59|-9.90069694519043
66|-9.89893283843994
57|-9.89654397964478
63|-9.88358898162842
61|-8.31776638031006



SELECT RunID, Max(TrainingAccuracy) FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60) GROUP BY RunID;
51|0.935620000469685
52|0.941539998687506
53|0.941839997528791
54|0.941879999707937
55|0.940640000852346
56|0.942040001772642
57|0.934479999608994
58|0.941399998310804
59|0.938660001899004
60|0.941099999366999





SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000 GROUP BY RunID;
51|-9.79214752214845
52|-9.79223536001058
53|-9.79208068583968
54|-9.80587766922328
55|-9.79199429324303
56|-9.80474778364331
57|-9.78004521111896
58|-9.79701188773682
59|-9.77236530951876
60|-9.7833963831135


SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration > 80000 GROUP BY RunID;
51|-9.91753535732146
52|-9.90522053011002
53|-9.89768090401926
54|-9.90887350882254
55|-9.89947455929172
56|-9.91534696394397
57|-9.88317625291886
58|-9.91197657738962
59|-9.88682746887208
60|-9.89927431844896

SELECT RunId,AVG(Value) FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60) AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration > 80000 GROUP BY RunID;
51|0.759800183215841
52|0.697619421612361
53|0.693133265917987
54|0.696433804669296
55|0.705919123739691
56|0.689220424523238
57|0.83317227376143
58|0.70525431344586
59|0.73974565807497
60|0.712861680980573

SELECT RunID, MIN(Value) FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID;
51|-9.93302793502808
52|-9.91686754226685
53|-9.90768098831177
54|-9.92163553237915
55|-9.91062927246094
56|-9.92644653320312
57|-9.89654397964478
58|-9.92368068695068
59|-9.90069694519043
60|-9.91252241134644
61|-8.31776638031006

SELECT RunID, AVG(Value) AS Score, COUNT(*) AS CNT FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration >= (58604-2500) AND Iteration <= 58604 GROUP BY RunID ORDER BY Score ASC;
54|-9.80955341512507|11
56|-9.7986214204268|11
53|-9.79520558443936|11
55|-9.78642922314731|11
58|-9.77820236032659|11
63|-9.77601292350075|11
52|-9.76999224749479|11
60|-9.76681964180686|11
51|-9.76091829646717|11
62|-9.75634000084617|11
59|-9.74249029159546|11
57|-9.73529397791082|11
61|-8.31776628494262|11



SELECT RunID, AVG(Value) AS Score, COUNT(*) AS CNT FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration >= (58604-2500) AND Iteration <= 58604 GROUP BY RunID ORDER BY Score ASC;
54|-9.60448572233126|11
53|-9.59665668077062|11
63|-9.57939725992631|11
56|-9.5765295665649|11
55|-9.57386443610536|11
58|-9.54747850598563|11
52|-9.54041633889051|11
60|-9.5363521717475|11
62|-9.5222587090034|11
57|-9.52161034493809|11
59|-9.50802985458516|11
51|-9.50126785603879|11
61|-8.31776617542048|11



SELECT RunID, AVG(Value) AS Score, COUNT(*) AS CNT FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63) AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration >= (58604-2500) AND Iteration <= 58604 GROUP BY RunID ORDER BY Score ASC;
61|0.337865308062597|11
60|1.12935366034508|11
53|1.14744955084541|11
52|1.16142424724319|11
59|1.16418322920799|11
62|1.16892394423485|11
56|1.17376239733262|11
58|1.18965362472968|11
54|1.19042155796831|11
55|1.2062313887206|11
63|1.22611752856861|11
57|1.24474751461636|11
51|1.25576961311427|11




SELECT RunID, AVG(Value) AS Score, COUNT(*) AS CNT FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63) AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration >= (58604-2500) AND Iteration <= 58604 GROUP BY RunID ORDER BY Score ASC;
61|0.45594413349049|11
53|1.42187191316502|11
60|1.43451765826549|11
63|1.43729895563426|11
56|1.4427888986131|11
55|1.46539479306102|11
54|1.46651505649864|11
58|1.46666475881672|11
52|1.48955829510662|11
62|1.49949010117377|11
59|1.50890852170884|11
51|1.59074708946561|11
57|1.68533579316785|11








SELECT RunID, AVG(TrainingAccuracy) AS Acc, COUNT(*) AS CNT FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63) AND Iteration >= (58604-2500) AND Iteration <= 58604 GROUP BY RunID ORDER BY Acc DESC;
53|0.874118181257681|11
63|0.873532727547451|11
56|0.871692726943709|11
60|0.871558181365078|11
55|0.869400000085289|11
54|0.869218181617477|11
58|0.868676363767819|11
59|0.866916363784833|11
52|0.866916363509785|11
62|0.866554545189251|11
51|0.858543636116656|11
57|0.856243637504686|11
61|0.84149818188754|11











SELECT RunID, AVG(TestAccuracy) AS Acc, COUNT(*) AS CNT FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63) AND Iteration >= (58604-2500) AND Iteration <= 58604 GROUP BY RunID ORDER BY Acc DESC;
60|0.907918181533705|11
53|0.907809090639786|11
63|0.907154545080662|11
52|0.90624545530731|11
56|0.90609090949893|11
54|0.905890909244798|11
59|0.905572727653113|11
62|0.905000001824444|11
58|0.904699999855865|11
57|0.902909091456912|11
55|0.902163635487448|11
51|0.899681816978346|11
61|0.891636363818429|11



SELECT RunID, MAX(TestAccuracy) AS Acc FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62) AND Iteration <= 76734 GROUP BY RunID ORDER BY Acc DESC;
SELECT RunID, MAX(TrainingAccuracy) AS Acc FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62) AND Iteration <= 51989 GROUP BY RunID ORDER BY Acc DESC;



SELECT RunID, Iteration, Value FROM run_kv_store WHERE RunID IN (54) AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration >= (23569-2500) AND Iteration <= 23569;
SELECT RunID, AVG(Value) AS Score, COUNT(*) AS CNT FROM run_kv_store WHERE RunID IN (54) AND Key LIKE "%classification_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" AND Iteration >= (23569-2500) AND Iteration <= 23569 GROUP BY RunID;


-- TUNA 2 - With Reverted Repo, Reverted before CBAM routing.
--Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With Random Augmentation - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62) GROUP BY RunID;
--SELECT * FROM logs_table WHERE RunID = 59;
--SELECT * FROM run_kv_store WHERE RunID = 61 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 51 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";
--SELECT RunID, Max(Epoch), Max(Iteration) FROM logs_table WHERE RunID IN (59) GROUP BY RunID;
--SELECT RunID, MIN(Value) FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID;

SELECT RunID, AVG(Value) AS Score, COUNT(*) AS CNT FROM run_kv_store WHERE RunID IN (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration >= (44149-2500) AND Iteration <= 44149 GROUP BY RunID ORDER BY Score ASC;
56|-9.49488489844582|11
58|-9.40937653454867|11
62|-9.37456755204634|11
55|-9.35274840268222|11
51|-9.34869000694968|11
57|-9.3281982161782|11
60|-9.32809641144492|11
59|-9.30455932617187|11
54|-9.2791233106093|11
52|-9.2192324551669|11
53|-9.02427030476657|11
61|-8.31776628494262|11
SELECT RunID, AVG(Value) AS Score, COUNT(*) AS CNT FROM run_kv_store WHERE RunID IN (59) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" AND Iteration >= (44149-2500) AND Iteration <= 44149 GROUP BY RunID ORDER BY Score ASC;
SELECT RunID, Min(Value) FROM run_kv_store WHERE RunID >= 59 AND RunID <= 67 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID;
59|-8.81324062347412
60|-9.90699548721313
61|-9.88932189941406
62|-9.90888805389404
63|-9.88592090606689
64|-9.90752983093262
65|-9.91554317474365
66|-9.90387783050537
67|-9.85255832672119



SELECT RunID, Max(TestAccuracy) As Acc FROM logs_table WHERE RunID >= 59 AND RunID <= 67 GROUP BY RunID ORDER BY Acc DESC;
65|0.93909999782443
60|0.938399994409084
64|0.937700003546476
66|0.935199998891354
61|0.934800001984835
63|0.934600001657009
59|0.933299997997284
62|0.932899998581409
67|0.926799997550249



SELECT RunID, Max(Epoch), Max(Iteration) FROM logs_table WHERE RunID IN (59, 60, 61) GROUP BY RunID;

--Experiments: "Gather Scatter Cigt With Focal Loss - Focal Loss Gamma: 2.0 - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--weight_decay = 5 * [0.0005]
--information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--warm_up_period = adjust_to_batch_size(original_value=350, target_batch_size=batch_size)
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = False
--ResnetCigtConstants.use_focal_loss = True
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 6/08/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With Focal Loss - Focal Loss Gamma: 2.0 - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (48, 49, 50, 51, 52) GROUP BY RunID;
--SELECT * FROM logs_table WHERE RunID = 48;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (48) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 48 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 48 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";
48|0.910899997633696
49|0.917499995410442
50|0.920499998056889
51|0.922100000995398
52|0.919599999964237




--Experiments: "Gather Scatter Cigt With Random Augmentation - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0006]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
--    {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [1.0, 1.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 6/11/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With Random Augmentation - [1,2,2] - [1.0, 1.0] - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (53, 54, 55, 56, 57) GROUP BY RunID;
--SELECT * FROM logs_table WHERE RunID = 51;
--SELECT * FROM run_kv_store WHERE RunID = 51 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT * FROM run_kv_store WHERE RunID = 51 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (53) GROUP BY RunID;

53|0.932599999552965
54|0.929699999302626
55|0.926300004637241
56|0.928599994152784
57|0.929400000858307




--Experiments: "Gather Scatter Cigt With CBAM Routers - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
--    {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 3
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = False
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 6/17/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger2.db"
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With CBAM Routers - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (80) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (80, 81, 82) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 58 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";

SELECT RunID, MIN(Value) FROM run_kv_store WHERE RunID IN (80, 81, 82, 83, 84) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID;
80|-9.94133834838867
81|-9.90903692245483
82|-9.92250785827637
83|-9.91958341598511
84|-9.91922626495361


SELECT RunID, MIN(Value) FROM run_kv_store WHERE RunID IN (80, 81, 82) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" GROUP BY RunID;
80|-10.2161219849878
81|-10.213179276914
82|-10.2093060551857


SELECT RunID, Max(TestAccuracy) AS Acc FROM logs_table WHERE RunID IN (80, 81, 82, 83, 84) GROUP BY RunID ORDER BY Acc DESC;
82|0.931699998658896
81|0.931199999392033
80|0.929199998182058
84|0.928999998366833
83|0.928800000786781


SELECT RunID, Max(Epoch), Max(Iteration) FROM logs_table WHERE RunID IN (80, 81, 82) GROUP BY RunID;



--Experiments: "Gather Scatter Cigt With CBAM Routers With Random Augmentation - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
--    {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 3
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 6/17/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With CBAM Routers With Random Augmentation - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (64) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (64, 65, 66, 67) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 64 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";

SELECT RunID, MIN(Value) FROM run_kv_store WHERE RunID IN (64, 65, 66, 67, 68) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID;
64|-9.9499119758606
65|-9.94210147857666
66|-9.94785537719727
67|-9.95525979995728
68|-9.94803733825684



SELECT RunID, MIN(Value) FROM run_kv_store WHERE RunID IN (64, 65, 66, 67) AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" GROUP BY RunID;
64|-9.86779645024514
65|-9.84381035396031
66|-9.72520600532999



SELECT RunID, Max(Epoch), Max(Iteration) FROM logs_table WHERE RunID IN (64, 65) GROUP BY RunID;

SELECT RunID, Max(TestAccuracy) AS Acc FROM logs_table WHERE RunID IN (64, 65, 66, 67, 68) GROUP BY RunID ORDER BY Acc DESC;
65|0.939399997836351
64|0.938100003147125
68|0.937199997973442
66|0.935999997842312
67|0.935600002139807



--Experiments: "Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:2  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
-------- {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 3
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.cbam_layer_input_reduction_ratio = 2
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 6/20/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:2  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (64) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (64, 65, 66, 67) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 64 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";

SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 145 AND RunID <= 154 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID ORDER BY Val ASC;
154|-9.95547275543213
148|-9.95539855957031
153|-9.94958620071411
152|-9.94492835998535
151|-9.94391832351685
150|-9.94099626541138
146|-9.93377876281738
145|-9.44920997619629
149|-9.41617317199707
147|-8.8265495300293


SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 145 AND RunID <= 154 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" GROUP BY RunID ORDER BY Val ASC;
154|-9.86252806138019
153|-9.8621167358087
151|-9.85688413892474
152|-9.85673902472671
148|-9.85374692021584
146|-9.85314713692179
150|-9.8438991235227
145|-9.39818103945985
149|-9.36693715076057
147|-8.77491552002576

SELECT RunID, Max(TestAccuracy) AS Acc FROM logs_table WHERE RunID >= 145 AND RunID <= 154 GROUP BY RunID ORDER BY Acc DESC;
153|0.938500000905991
148|0.937199995589256
152|0.936999999809265
151|0.936599998688698
149|0.935399998909235
150|0.934900001716614
145|0.934700001239777
154|0.934299994570017
146|0.932400004458427
147|0.929799998188019

--X3
--Experiments: "Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
-------- {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 3
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.cbam_layer_input_reduction_ratio = 4
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 6/20/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (64) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (64, 65, 66, 67) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 64 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";

SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID >= 94 AND RunID <= 107 GROUP BY RunID;

SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 94 AND RunID <= 124 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID ORDER BY Val ASC;
100|-9.95209856033325
99|-9.95058145523071
98|-9.95023164749145
94|-9.94882926940918
118|-9.93463611602783
109|-9.93376703262329
108|-9.93076553344727
107|-9.93050956726074
114|-9.44133768081665
115|-9.44051666259766
106|-9.43547973632812
103|-9.43381786346436
116|-9.43366775512695
97|-9.43189783096313
113|-9.43132076263428
101|-9.43078145980835
112|-9.42592363357544
95|-9.42155113220215
111|-9.41837949752808
110|-9.41765975952148
96|-8.83351440429687
105|-8.82539157867432
104|-8.81115369796753
117|-8.31779403686523
102|-8.31776723861694





SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 94 AND RunID <= 109 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" GROUP BY RunID ORDER BY Val ASC;
100|-9.95209856033325
99|-9.95058145523071
98|-9.95023164749145
94|-9.94882926940918
109|-9.93376703262329
108|-9.93076553344727
107|-9.93050956726074
106|-9.43547973632812
103|-9.43381786346436
97|-9.43189783096313
101|-9.43078145980835
95|-9.42155113220215
110|-9.40627937316895
96|-8.83351440429687
105|-8.82539157867432
104|-8.81115369796753
102|-8.31776723861694

SELECT RunID, Epoch, Max(TestAccuracy) AS Acc FROM logs_table WHERE RunID IN (100, 99, 98, 94, 109, 108, 107, 118, 122, 123, 124) GROUP BY RunID ORDER BY Acc DESC;
94|1390|0.937399998956919
98|1740|0.93649999628067
99|1425|0.938099999165535
100|1747|0.936699999719858
107|1480|0.937499996352196
108|1525|0.935499993807077
109|1650|0.937800004923344
118|1465|0.937699996739626
122|1745|0.938899998319149
123|1710|0.939400000721216



SELECT RunID, Max(TrainingAccuracy) AS Acc FROM logs_table WHERE RunID IN (100, 99, 98, 94, 109, 108, 107) GROUP BY RunID ORDER BY RunID ASC;
94|0.940020001798868
98|0.940859999191761
99|0.942979999703169
100|0.940099999934435
107|0.940999999938011
108|0.939420000171661
109|0.942540000211


SELECT RunID, Max(TestAccuracy) AS Acc FROM logs_table WHERE RunID >= 94 AND RunID <= 122 GROUP BY RunID ORDER BY Acc DESC;
99|0.938099999165535
113|0.937999995720387
109|0.937800004923344
97|0.937500004208088
107|0.937499996352196
94|0.937399998956919
114|0.937199995708466
111|0.936800005698204
100|0.936699999719858
98|0.93649999628067
115|0.936399997407198
105|0.936199997901917
103|0.935900000649691
108|0.935499993807077
95|0.934800002795458
106|0.934299995917082
112|0.933599997448921
116|0.933400005424023
101|0.9332000007689
110|0.932700000917911
96|0.931899998092651
102|0.930700000190735
117|0.930600000476837
118|0.923200002288818
104|0.832100001186132


--X2
--Experiments: "Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:0  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:12 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
-------- {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 12
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.cbam_layer_input_reduction_ratio = 0
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 6/24/2023
--Started on: HPC - "/clusterusers/can.bicici@boun.edu.tr/cigt/cigt/dblogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:0  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:12 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (155, 156) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (155, 156, 157, 158) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 64 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";

SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 155 AND RunID <= 170 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID ORDER BY Val ASC;
166|-9.98658447265625
156|-9.98201055526733
165|-9.98135442733765
168|-9.9798713684082
157|-9.97910108566284
170|-9.97743701934814
167|-9.97512502670288
160|-9.97467069625855
161|-9.97393388748169
164|-9.97010927200317
169|-9.97003049850464
162|-9.96883096694946
158|-9.96735868453979
163|-9.96383390426636
159|-9.96234474182129
155|-8.79235649108887






SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 155 AND RunID <= 163 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" GROUP BY RunID ORDER BY Val ASC;
156|-9.92062385714784
157|-9.8983810775134
159|-9.89207063402448
158|-9.89066238792575
160|-9.07284777505057
155|-8.71533734457833


SELECT RunID, Max(TestAccuracy) AS acc FROM logs_table WHERE RunID >= 155 AND RunID <= 170 GROUP BY RunID ORDER BY acc DESC;
162|0.940300001215935
168|0.940000001740456
170|0.939999999791384
158|0.9396000028193
165|0.939399999284744
156|0.938900003194809
164|0.938700002789497
160|0.938300004941225
163|0.938299999415874
167|0.938199998360872
166|0.938099997085333
169|0.938000001436472
161|0.937800001263619
159|0.936200001358986
157|0.935900003427267
155|0.796900004029274





SELECT RunID, Max(TrainingAccuracy) AS acc FROM logs_table WHERE RunID >= 155 AND RunID <= 163 GROUP BY RunID ORDER BY acc DESC;
156|0.945639996863604
159|0.943559998449087
157|0.943359999513626
162|0.942759997390509
158|0.942259998605251
160|0.94200000069499
161|0.94016000030756
163|0.889480002545118
155|0.67955999712348




SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID >= 155 AND RunID <= 159 GROUP BY RunID;



--Experiments: "Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:0  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:6 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
-------- {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 6
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.cbam_layer_input_reduction_ratio = 0
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 6/27/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:0  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:6 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (85, 86) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (155, 156, 157, 158) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 64 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";

SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID >= 85 AND RunID <= 93 GROUP BY RunID;
85|0.9373000005126
86|0.937600005996227
87|0.937800000333786
88|0.939699998509884
89|0.940299997723103
90|0.938299998652935
91|0.93869999987483
92|0.941099996399879
93|0.853199997657538
SELECT Max(TestAccuracy) FROM logs_table WHERE RunID >= 85 AND RunID <= 93 GROUP BY RunID;
0.9373000005126
0.937600005996227
0.937800000333786
0.939699998509884
0.940299997723103
0.938299998652935
0.93869999987483
0.941099996399879
0.853199997657538


SELECT RunID, Max(TrainingAccuracy) FROM logs_table WHERE RunID >= 85 AND RunID <= 93 GROUP BY RunID;
85|0.942960001524687
86|0.942500000519752
87|0.940279998354912
88|0.940679999935627
89|0.941440001132488
90|0.94398000010848
91|0.942620001559258
92|0.943520000168085
93|0.743840003054142
SELECT Max(TrainingAccuracy) FROM logs_table WHERE RunID >= 85 AND RunID <= 93 GROUP BY RunID;
0.942960001524687
0.942500000519752
0.940279998354912
0.940679999935627
0.941440001132488
0.94398000010848
0.942620001559258
0.943520000168085
0.743840003054142



SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 85 AND RunID <= 93 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID ORDER BY Val ASC;
86|-9.97135543823242
92|-9.97055788040161
90|-9.97016839981079
87|-9.96996917724609
85|-9.96827507019043
91|-9.96748495101929
88|-9.9650918006897
89|-9.96475257873535
93|-9.28416738510132



SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 85 AND RunID <= 86 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%" GROUP BY RunID ORDER BY Val ASC;
85|-9.89714110627466
86|-9.73989130526173


--Experiments: "Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0006]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
-------- {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 3
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.cbam_layer_input_reduction_ratio = 4
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 7/02/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (64) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (64, 65, 66, 67) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 64 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";
SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID >= 94 AND RunID <= 103 GROUP BY RunID;
94|0.928399998188019
95|0.936300002419949
96|0.927800000572205
97|0.930299997997284
98|0.937199999511242
99|0.930200000286102
100|0.931899997806549
101|0.927899998283386
102|0.925899998855591
103|0.786100001049042




SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 94 AND RunID <= 103 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID ORDER BY Val ASC;
98|-9.94574613571167
95|-9.4231499671936
94|-8.31789779663086
103|-8.31777667999268
100|-8.3177698135376
96|-8.31776943206787
97|-8.31776647567749
99|-8.31776647567749
101|-8.31776647567749
102|-8.31776638031006



--Experiments: "Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0004 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0004]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
-------- {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 3
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.cbam_layer_input_reduction_ratio = 4
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--Started at 7/05/2023
--Started on: Tetam - "/cta/users/ucbicici/cigt/cigt/cigtlogger2.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0004 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (64) GROUP BY RunID;
--SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID IN (64, 65, 66, 67) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 64 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%";

SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID >= 71 AND RunID <= 76 GROUP BY RunID;
71|0.935799999862909
72|0.935700000846386
73|0.938400002968311




SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 71 AND RunID <= 73 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID ORDER BY Val ASC;
71|0.935799999862909
72|0.935700000846386
73|0.939299999982119
74|0.939600001823902
75|0.938400000905991
76|0.931900000286102




--Experiments: "Vanilla With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:0  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:6 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization"
--weight_decay = 5 * [0.0005]
--ResnetCigtConstants.resnet_config_list = [
--    {"path_count": 1,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
--    {"path_count": 2,
--     "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
-------- {"layer_count": 18, "feature_map_count": 16}]},
--    {"path_count": 4,
--     "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
--ResnetCigtConstants.classification_wd = param_tpl[0]
--ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
--ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
--ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
--ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
--ResnetCigtConstants.decision_drop_probability = 0.5
--ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 6
--ResnetCigtConstants.cbam_reduction_ratio = 4
--ResnetCigtConstants.cbam_layer_input_reduction_ratio = 0
--ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
--ResnetCigtConstants.decision_dimensions = [128, 128]
--ResnetCigtConstants.apply_mask_to_batch_norm = False
--ResnetCigtConstants.advanced_augmentation = True
--ResnetCigtConstants.use_focal_loss = False
--ResnetCigtConstants.focal_loss_gamma = 2.0
--ResnetCigtConstants.batch_norm_type = "BatchNorm"
--ResnetCigtConstants.data_parallelism = False
--Started at 7/05/2023
--Started on: Tetam - "/cta/users/hmeral/cigt/cigt/cigtlogger.db"
--SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Vanilla With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:0  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:6 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization%";
SELECT RunID, Max(Epoch) FROM logs_table WHERE RunID >= 75 AND RunID <= 78 GROUP BY RunID;

SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID >= 75 AND RunID <= 80 GROUP BY RunID;
75|0.93820000371933
76|0.939700002145767
77|0.937300003159046
78|0.937600000214577
79|0.939199999707937
80|0.937700006979704




SELECT RunID, MIN(Value) AS Val FROM run_kv_store WHERE RunID >= 75 AND
RunID <= 80 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%test%" GROUP BY RunID ORDER BY Val ASC;
79|-9.98696050643921
77|-9.98210973739624
76|-9.97394971847534
78|-9.97310676574707
80|-9.96806879043579
75|-9.96447162628174

