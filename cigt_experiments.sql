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
61|0.999971600027466|0.927533999092102|0.00075|1.0|1.0|0.999899999999999|0.1|50
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
30|0.877400002098083



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
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (95,96) GROUP BY RunID;
95|0.934000000286102
96|0.929099997997284


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
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (29,30,31) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 31 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
30|0.933299997711182
31|0.923999998766184


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
--SELECT RunID, Max(TestAccuracy) FROM logs_table WHERE RunID IN (23,24) GROUP BY RunID;
--SELECT * FROM run_kv_store WHERE RunID = 23 AND Key LIKE "%routing_loss%" AND Key NOT LIKE "%Layer%" AND Key LIKE "%train%";
--SELECT Epoch,TestAccuracy FROM logs_table WHERE RunID = 23 ORDER BY TestAccuracy DESC;
23|0.925400000804663 (1390) randig_cigtlogger2_23_epoch1390




