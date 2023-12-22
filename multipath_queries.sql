SELECT RunId, Epoch, Max(TestAccuracy) AS TestAccuracy, TestMac, Wd, MacLambda FROM
(
	SELECT * FROM logs_table_q_cigt
	LEFT JOIN (SELECT RunId, Value AS Wd FROM run_parameters WHERE Parameter = 'policyNetworksWd') AS A
	ON logs_table_q_cigt.RunId = A.RunId
	LEFT JOIN (SELECT RunId, Value AS MacLambda FROM run_parameters WHERE Parameter = 'policyNetworksMacLambda') AS B
	ON logs_table_q_cigt.RunId = B.RunId
	WHERE 
		logs_table_q_cigt.RunId IN 
		(SELECT RunId FROM run_kv_store WHERE Key = 'Training Status' AND Value = 'Training Finished!!!') AND 
		logs_table_q_cigt.Epoch > -1
)
GROUP BY RunId
ORDER BY TestAccuracy DESC


SELECT Avg(TestAccuracy) AS TestAccuracy, Avg(TestMac) AS TestMac, Wd, MacLambda, COUNT(*) AS CNT
FROM
(
	SELECT RunId, Epoch, TestAccuracy, TestMac, Wd, MacLambda FROM
	(
		SELECT * FROM logs_table_q_cigt
		LEFT JOIN (SELECT RunId, Value AS Wd FROM run_parameters WHERE Parameter = 'policyNetworksWd') AS A
		ON logs_table_q_cigt.RunId = A.RunId
		LEFT JOIN (SELECT RunId, Value AS MacLambda FROM run_parameters WHERE Parameter = 'policyNetworksMacLambda') AS B
		ON logs_table_q_cigt.RunId = B.RunId
		WHERE
			logs_table_q_cigt.RunId IN
			(SELECT RunId FROM run_kv_store WHERE Key = 'Training Status' AND Value = 'Training Finished!!!') AND
			logs_table_q_cigt.Epoch > -1
	)
	WHERE TestAccuracy > 0.9415
	ORDER BY TestAccuracy DESC
)
GROUP BY Wd, MacLambda
ORDER BY TestAccuracy DESC