#!/usr/bin/python3

import numpy as np
from glob import glob

#setting
raw_dir = 'raw'
pass_keyword = "Can't get"
awalan = '<PRE>'
akhiran = '</PRE>'

all_file = glob('%s/*'%(raw_dir))

#gather data
dataset_complete = []
for i in range(len(all_file)):
	print('progress %s/%s...'%(i+1, len(all_file)))
	alldata_open = open(all_file[i])
	alldata = alldata_open.read()
	
	if pass_keyword in alldata:
		alldata_open.close()
		print('data empty, continue...')
		continue
	
	alldata = alldata.split('\n')
	awalan_idx = -1
	for j in range(len(alldata)):
		if awalan in  alldata[j]:
			awalan_idx = j
			break
	if awalan_idx == -1:
		alldata_open.close()
		print('data empty, continue...')
		continue
	
	akhiran_idx = -1
	for j in range(awalan_idx, len(alldata)):
		if akhiran in alldata[j]:
			akhiran_idx = j
			break
	if akhiran_idx == -1:
		alldata_open.close()
		print('data empty, continue...')
		continue
	
	if akhiran_idx-awalan_idx < 6:
		alldata_open.close()
		print('data empty, continue...')
		continue
	
	for j in range(awalan_idx+5, akhiran_idx):
		if len(alldata[j]) < 44:
			continue
		try:
			tekanan = float(alldata[j][:7])
			suhu = float(alldata[j][7*2:7*3])
			rh = float(alldata[j][7*4:7*5])
			mixing_ratio = float(alldata[j][7*5:7*6])
			dataset_complete.append([tekanan, suhu, rh, mixing_ratio])
		except ValueError:
			continue

dataset_complete = np.array(dataset_complete)
print(np.shape(dataset_complete))

#save
np.save('dataset.npy', dataset_complete)
