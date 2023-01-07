import ordered_feature_selector
import loocv_accuracy
import feature_importances_list

print(loocv_accuracy.find_accuracy("_greedy_final.csv"))
# names = ['Sex','Hispanic','Diabetes','Hypertension','Neutrophils','Monocytes','B cells','MHCII Monocytes','Resistin','IL-8','IP-10','IL-6','IFNλ2/3','Platelets','MHCII+Platelets','LCN2','Myoglobin','CRP','OPN','MPO','ICAM-1','VCAM-1','Cystatin C','D-dimer']
# print(feature_importances_list.avg_feature_importances("_imputed_data.csv", names, 10000))