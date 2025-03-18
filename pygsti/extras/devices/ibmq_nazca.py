""" Specification of IBM Q Nazca """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


qubits = ['Q' + str(x) for x in range(127)]

two_qubit_gate = 'Gcnot'

edgelist = [
    # 1st row of connections
    ('Q0', 'Q1'), ('Q1', 'Q0'),
    ('Q1', 'Q2'), ('Q2', 'Q1'),
    ('Q2', 'Q3'), ('Q3', 'Q2'),
    ('Q3', 'Q4'), ('Q4', 'Q3'),
    ('Q4', 'Q5'), ('Q5', 'Q4'),
    ('Q5', 'Q6'), ('Q6', 'Q5'),
    ('Q6', 'Q7'), ('Q7', 'Q6'),
    ('Q7', 'Q8'), ('Q8', 'Q7'),
    ('Q8', 'Q9'), ('Q9', 'Q8'),
    ('Q9', 'Q10'), ('Q10', 'Q9'),
    ('Q10', 'Q11'), ('Q11', 'Q10'),
    ('Q11', 'Q12'), ('Q12', 'Q11'),
    ('Q12', 'Q13'), ('Q13', 'Q12'),
    # 2nd row of connections
    ('Q18', 'Q19'), ('Q19', 'Q18'),
    ('Q19', 'Q20'), ('Q20', 'Q19'),
    ('Q20', 'Q21'), ('Q21', 'Q20'),
    ('Q21', 'Q22'), ('Q22', 'Q21'),
    ('Q22', 'Q23'), ('Q23', 'Q22'),
    ('Q23', 'Q24'), ('Q24', 'Q23'),
    ('Q24', 'Q25'), ('Q25', 'Q24'),
    ('Q25', 'Q26'), ('Q26', 'Q25'),
    ('Q26', 'Q27'), ('Q27', 'Q26'),
    ('Q27', 'Q28'), ('Q28', 'Q27'),
    ('Q28', 'Q29'), ('Q29', 'Q28'),
    ('Q29', 'Q30'), ('Q30', 'Q29'),
    ('Q30', 'Q31'), ('Q31', 'Q30'),
    ('Q31', 'Q32'), ('Q32', 'Q31'),
    # 3rd row of connections
    ('Q37', 'Q38'), ('Q38', 'Q37'),
    ('Q38', 'Q39'), ('Q39', 'Q38'),
    ('Q39', 'Q40'), ('Q40', 'Q39'),
    ('Q40', 'Q41'), ('Q41', 'Q40'),
    ('Q41', 'Q42'), ('Q42', 'Q41'),
    ('Q42', 'Q43'), ('Q43', 'Q42'),
    ('Q43', 'Q44'), ('Q44', 'Q43'),
    ('Q44', 'Q45'), ('Q45', 'Q44'),
    ('Q45', 'Q46'), ('Q46', 'Q45'),
    ('Q46', 'Q47'), ('Q47', 'Q46'),
    ('Q47', 'Q48'), ('Q48', 'Q47'),
    ('Q48', 'Q49'), ('Q49', 'Q48'),
    ('Q49', 'Q50'), ('Q50', 'Q49'),
    ('Q50', 'Q51'), ('Q51', 'Q50'),
    # 4th row of connections
    ('Q56', 'Q57'), ('Q57', 'Q56'),
    ('Q57', 'Q58'), ('Q58', 'Q57'),
    ('Q58', 'Q59'), ('Q59', 'Q58'),
    ('Q59', 'Q60'), ('Q60', 'Q59'),
    ('Q60', 'Q61'), ('Q61', 'Q60'),
    ('Q61', 'Q62'), ('Q62', 'Q61'),
    ('Q62', 'Q63'), ('Q63', 'Q62'),
    ('Q63', 'Q64'), ('Q64', 'Q63'),
    ('Q64', 'Q65'), ('Q65', 'Q64'),
    ('Q65', 'Q66'), ('Q66', 'Q65'),
    ('Q66', 'Q67'), ('Q67', 'Q66'),
    ('Q67', 'Q68'), ('Q68', 'Q67'),
    ('Q68', 'Q69'), ('Q69', 'Q68'),
    ('Q69', 'Q70'), ('Q70', 'Q69'),
    # 5th row of connections
    ('Q75', 'Q76'), ('Q76', 'Q75'), 
    ('Q76', 'Q77'), ('Q77', 'Q76'),
    ('Q77', 'Q78'), ('Q78', 'Q77'),
    ('Q78', 'Q79'), ('Q79', 'Q78'),
    ('Q79', 'Q80'), ('Q80', 'Q79'),
    ('Q80', 'Q81'), ('Q81', 'Q80'),
    ('Q81', 'Q82'), ('Q82', 'Q81'),
    ('Q82', 'Q83'), ('Q83', 'Q82'),
    ('Q83', 'Q84'), ('Q84', 'Q83'),
    ('Q84', 'Q85'), ('Q85', 'Q84'),
    ('Q85', 'Q86'), ('Q86', 'Q85'),
    ('Q86', 'Q87'), ('Q87', 'Q86'),
    ('Q87', 'Q88'), ('Q88', 'Q87'),
    ('Q88', 'Q89'), ('Q89', 'Q88'),
    # 6th row of connections
    ('Q94', 'Q95'), ('Q95', 'Q94'),
    ('Q95', 'Q96'), ('Q96', 'Q95'), 
    ('Q96', 'Q97'), ('Q97', 'Q96'),
    ('Q97', 'Q98'), ('Q98', 'Q97'),
    ('Q98', 'Q99'), ('Q99', 'Q98'),
    ('Q99', 'Q100'), ('Q100', 'Q99'),
    ('Q100', 'Q101'), ('Q101', 'Q100'),
    ('Q101', 'Q102'), ('Q102', 'Q101'),
    ('Q102', 'Q103'), ('Q103', 'Q102'),
    ('Q103', 'Q104'), ('Q104', 'Q103'),
    ('Q104', 'Q105'), ('Q105', 'Q104'),
    ('Q105', 'Q106'), ('Q106', 'Q105'),
    ('Q106', 'Q107'), ('Q107', 'Q106'),
    ('Q107', 'Q108'), ('Q108', 'Q107'),
    # 7th row of connections
    ('Q113', 'Q114'), ('Q114', 'Q113'),
    ('Q114', 'Q115'), ('Q115', 'Q114'),
    ('Q115', 'Q116'), ('Q116', 'Q115'),
    ('Q116', 'Q117'), ('Q117', 'Q116'),
    ('Q117', 'Q118'), ('Q118', 'Q117'),
    ('Q118', 'Q119'), ('Q119', 'Q118'),
    ('Q119', 'Q120'), ('Q120', 'Q119'),
    ('Q120', 'Q121'), ('Q121', 'Q120'),
    ('Q121', 'Q122'), ('Q122', 'Q121'),
    ('Q122', 'Q123'), ('Q123', 'Q122'),
    ('Q123', 'Q124'), ('Q124', 'Q123'),
    ('Q124', 'Q125'), ('Q125', 'Q124'),
    ('Q125', 'Q126'), ('Q126', 'Q125'),
    # 1st column of connections
    ('Q0', 'Q14'), ('Q14', 'Q0'),
    ('Q14', 'Q18'), ('Q18', 'Q14'),
    ('Q37', 'Q52'), ('Q52', 'Q37'),
    ('Q52', 'Q56'), ('Q56', 'Q52'),
    ('Q75', 'Q90'), ('Q90', 'Q75'),
    ('Q90', 'Q94'), ('Q94', 'Q90'),
    # 2nd column of connections
    ('Q20', 'Q33'), ('Q33', 'Q20'),
    ('Q33', 'Q39'), ('Q39', 'Q33'),
    ('Q58', 'Q71'), ('Q71', 'Q58'),
    ('Q71', 'Q77'), ('Q77', 'Q71'),
    ('Q96', 'Q109'), ('Q109', 'Q96'),
    ('Q109', 'Q114'), ('Q114', 'Q109'),
    # 3rd column of connections
    ('Q4', 'Q15'), ('Q15', 'Q4'),
    ('Q15', 'Q22'), ('Q22', 'Q15'),
    ('Q41', 'Q53'), ('Q53', 'Q41'),
    ('Q53', 'Q60'), ('Q60', 'Q53'),
    ('Q79', 'Q91'), ('Q91', 'Q79'),
    ('Q91', 'Q98'), ('Q98', 'Q91'),
    # 4th column of connections
    ('Q24', 'Q34'), ('Q34', 'Q24'),
    ('Q34', 'Q43'), ('Q43', 'Q34'),
    ('Q62', 'Q72'), ('Q72', 'Q62'),
    ('Q72', 'Q81'), ('Q81', 'Q72'),
    ('Q100', 'Q110'), ('Q110', 'Q100'),
    ('Q110', 'Q118'), ('Q118', 'Q110'),
    # 5th column of connections
    ('Q8', 'Q16'), ('Q16', 'Q8'),
    ('Q16', 'Q26'), ('Q26', 'Q16'),
    ('Q45', 'Q54'), ('Q54', 'Q45'),
    ('Q54', 'Q64'), ('Q64', 'Q54'),
    ('Q83', 'Q92'), ('Q92', 'Q83'),
    ('Q92', 'Q102'), ('Q102', 'Q92'),
    # 6th column of connections
    ('Q28', 'Q35'), ('Q35', 'Q28'),
    ('Q35', 'Q47'), ('Q47', 'Q35'),
    ('Q66', 'Q73'), ('Q73', 'Q66'),
    ('Q73', 'Q85'), ('Q85', 'Q73'),
    ('Q104', 'Q111'), ('Q111', 'Q104'),
    ('Q111', 'Q122'), ('Q122', 'Q111'),
    # 7th column of connections
    ('Q12', 'Q17'), ('Q17', 'Q12'),
    ('Q17', 'Q30'), ('Q30', 'Q17'),
    ('Q49', 'Q55'), ('Q55', 'Q49'),
    ('Q55', 'Q68'), ('Q68', 'Q55'),
    ('Q87', 'Q93'), ('Q93', 'Q87'),
    ('Q93', 'Q106'), ('Q106', 'Q93'),
    # 8th column of connections
    ('Q32', 'Q36'), ('Q36', 'Q32'),
    ('Q36', 'Q51'), ('Q51', 'Q36'),
    ('Q70', 'Q74'), ('Q74', 'Q70'),
    ('Q74', 'Q89'), ('Q89', 'Q74'),
    ('Q108', 'Q112'), ('Q112', 'Q108'),
    ('Q112', 'Q126'), ('Q126', 'Q112'),
]

spec_format= 'ibmq_v2019'